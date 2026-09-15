"""
Amazon Robotics Hackathon - Routing API

Coordinated routing for weighted aisles, shared pickups, and limited docks.

*****IMPORTANT*****
Team name:
Email address:
*******************
"""

from functools import lru_cache
from collections import OrderedDict, deque
import heapq
import math
from time import perf_counter
from typing import Optional
from ar_hackathon.models.graph_state import GraphState


_INF = float("inf")


class _Layout:
    """Only immutable floor geometry is cached between calls or games."""

    def __init__(self, nodes, edges):
        self.nodes = {node[0]: node for node in nodes}
        self.edges = edges
        self.adj = {node: {} for node in self.nodes}
        for index, (source, destination, weight, capacity, both) in enumerate(edges):
            if source not in self.adj or destination not in self.adj:
                continue
            # get_edge uses the FIRST matching edge, including reverse matches.
            self.adj[source].setdefault(destination, index)
            if both:
                self.adj[destination].setdefault(source, index)
        self.distances = {}

    def distances_from(self, source):
        if source not in self.distances:
            result = {source: 0}
            queue = [(0, source)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance != result[node]:
                    continue
                for neighbor, index in self.adj.get(node, {}).items():
                    candidate = distance + self.edges[index][2]
                    if candidate < result.get(neighbor, _INF):
                        result[neighbor] = candidate
                        heapq.heappush(queue, (candidate, neighbor))
            self.distances[source] = result
        return self.distances[source]

    def distance(self, source, destination):
        return self.distances_from(source).get(destination, _INF)

    def can_visit(self, source, destinations):
        """A single load must fit a chain in the directed reachability graph."""
        destinations = tuple(set(destinations))
        if any(self.distance(source, node) == _INF for node in destinations):
            return False
        return all(self.distance(a, b) < _INF or self.distance(b, a) < _INF
                   for index, a in enumerate(destinations) for b in destinations[index + 1:])

    def first_step(self, source, destination):
        if source == destination:
            return None
        choices = [(self.edges[index][2] + self.distance(neighbor, destination),
                    neighbor)
                   for neighbor, index in self.adj.get(source, {}).items()]
        distance, neighbor = min(choices, default=(_INF, None))
        return neighbor if distance < _INF else None


@lru_cache(maxsize=4)
def _layout(nodes, edges):
    return _Layout(nodes, edges)


@lru_cache(maxsize=32)
def _joint_reachability(layout, positions):
    """Find physical reachability on tiny floors with no unlimited parking.

    On a narrow tree, robots cannot necessarily pass each other, even though
    every node is connected. Serial legal moves give an exact reachability
    check for resting robots; using inbound destinations is an optimistic
    check until transit completes. The caller bounds the configuration space.
    """
    reachable = [set([position]) for position in positions]
    seen = {positions}
    queue = deque([positions])
    while queue:
        configuration = queue.popleft()
        occupied = {}
        for node in configuration:
            occupied[node] = occupied.get(node, 0) + 1
        for index, node in enumerate(configuration):
            for neighbor in layout.adj[node]:
                if occupied.get(neighbor, 0) >= layout.nodes[neighbor][2]:
                    continue
                moved = configuration[:index] + (neighbor,) + configuration[index + 1:]
                if moved in seen:
                    continue
                seen.add(moved)
                reachable[index].add(neighbor)
                queue.append(moved)
        if all(len(nodes) == len(layout.nodes) for nodes in reachable):
            break
    return tuple(frozenset(nodes) for nodes in reachable)


_cooperative_cache = OrderedDict()
_COOPERATIVE_MISS = object()
_traffic_episodes = OrderedDict()


def _recover_if_stalled(state, layout, drive_unit_id):
    """Keep normal traffic parallel; coordinate only after a repeated state."""
    units = sorted(state.drive_units, key=lambda unit: unit.id)
    if (not 1 < len(units) <= 4 or len(layout.nodes) > 12 or
            not 0 < len(state.active_pods) <= 12 or
            any(node[2] is None for node in layout.nodes.values())):
        return False, None
    episode_key = layout, tuple((unit.id, unit.capacity) for unit in units)
    now = state.current_time_step
    episode = _traffic_episodes.get(episode_key)
    if (episode is None or now < episode["time"] or
            (now == episode["time"] and drive_unit_id <= episode["last_unit"])):
        episode = {"time": -1, "last_unit": drive_unit_id, "seen": OrderedDict(),
                   "active": False, "delivered": len(state.delivered_pods)}
        _traffic_episodes[episode_key] = episode
        while len(_traffic_episodes) > 4:
            _traffic_episodes.popitem(last=False)
    _traffic_episodes.move_to_end(episode_key)
    episode["last_unit"] = drive_unit_id
    # Later calls in this tick include earlier committed moves. Sample once
    # per tick so ordinary polling cannot look like a repeated waiting state.
    if now != episode["time"]:
        episode["time"] = now
        delivered = len(state.delivered_pods)
        if delivered != episode["delivered"]:
            episode["active"] = False
            episode["delivered"] = delivered
            episode["seen"].clear()
        signature = (
            tuple((unit.current_node, unit.transit_destination, unit.transit_remaining_time,
                   tuple(sorted(unit.carrying))) for unit in units),
            tuple(sorted((pod.id, pod.current_node, pod.carried_by,
                          pod.destination_station, pod.entry_time) for pod in state.active_pods)))
        if signature in episode["seen"]:
            episode["active"] = True
        episode["seen"][signature] = now
        while len(episode["seen"]) > 128:
            episode["seen"].popitem(last=False)
    if episode["active"]:
        handled, move = _cooperative_next_move(state, layout, drive_unit_id)
        if handled:
            return True, move
        episode["active"] = False
    return False, None


def _cooperative_next_move(state, layout, drive_unit_id):
    """Return (handled, move) for small floors with finite node capacity.

    Planning serial moves is conservative but permits exact capacity checking:
    only one robot moves at a time, and every destination slot is reserved.
    The search includes automatic FIFO pickup and delivery, so it can make an
    empty robot retreat several nodes before a loaded robot enters a dead end.
    Only immutable geometry and fully specified resting states are cached.
    """
    units = sorted(state.drive_units, key=lambda u: u.id)
    if (not state.active_pods or len(state.active_pods) > 12 or
            not 1 < len(units) <= 4 or
            len(layout.nodes) > 12 or
            any(node[2] is None for node in layout.nodes.values())):
        return False, None
    if any(unit.in_transit for unit in units):
        return True, None
    pods = sorted(state.active_pods, key=lambda p: (p.entry_time, p.id))
    ids = tuple(unit.id for unit in units)
    capacities = tuple(unit.capacity for unit in units)
    uid_index = {uid: i for i, uid in enumerate(ids)}
    sources = tuple(pod.current_node for pod in pods)
    destinations = tuple(pod.destination_station for pod in pods)
    metadata = tuple((pod.id, pod.destination_station, pod.entry_time) for pod in pods)
    initial = (tuple(unit.current_node for unit in units),
               tuple(-1 if pod.carried_by is None else uid_index[pod.carried_by]
                     for pod in pods))
    if any(node not in layout.nodes for node in initial[0]):
        return False, None

    def key(configuration):
        positions, status = configuration
        remaining = tuple((metadata[p], sources[p] if owner == -1 else None,
                           None if owner == -1 else ids[owner])
                          for p, owner in enumerate(status) if owner != -2)
        return layout, ids, capacities, positions, remaining

    def remember(cache_key, action):
        _cooperative_cache[cache_key] = action
        _cooperative_cache.move_to_end(cache_key)
        while len(_cooperative_cache) > 2048:
            _cooperative_cache.popitem(last=False)

    initial_key = key(initial)
    action = _cooperative_cache.get(initial_key, _COOPERATIVE_MISS)
    if action is not _COOPERATIVE_MISS:
        if action is None:
            return False, None
        return True, action[1] if action[0] == drive_unit_id else None

    node_capacities = {node: values[2] for node, values in layout.nodes.items()}
    adjacency = {node: tuple((neighbor, layout.edges[index][2])
                            for neighbor, index in neighbors.items()
                            if neighbor != node)
                 for node, neighbors in layout.adj.items()}
    distances = {node: layout.distances_from(node) for node in layout.nodes}
    infinity = float('inf')

    def settle(positions, status):
        updated = list(status)
        for u, node in enumerate(positions):
            for p, owner in enumerate(updated):
                if owner == u and destinations[p] == node:
                    updated[p] = -2
            free = capacities[u] - updated.count(u)
            if free:
                for p, owner in enumerate(updated):
                    if free and owner == -1 and sources[p] == node:
                        updated[p] = u
                        free -= 1
        return tuple(updated)

    def heuristic(configuration):
        positions, status = configuration
        longest = 0
        for p, owner in enumerate(status):
            if owner == -2:
                continue
            if owner >= 0:
                distance = distances[positions[owner]].get(destinations[p], infinity)
            else:
                distance = min(distances[node].get(sources[p], infinity)
                               for node in positions)
                distance += distances.get(sources[p], {}).get(destinations[p], infinity)
            # Unreachable jobs must not hide attainable partial deliveries.
            if distance != infinity:
                longest = max(longest, distance)
        return longest

    started = perf_counter()
    costs = {initial: 0}
    previous = {}
    queue = [(heuristic(initial), 0, 0, 0, initial)]
    serial = 0
    best = None
    best_rank = (0, infinity)
    expansions = 0
    while queue and expansions < 18000:
        if expansions % 32 == 0 and perf_counter() - started > 0.18:
            break
        _, _, elapsed, _, configuration = heapq.heappop(queue)
        if costs.get(configuration) != elapsed:
            continue
        positions, status = configuration
        delivered = status.count(-2)
        rank = (-delivered, elapsed)
        if delivered and rank < best_rank:
            best, best_rank = configuration, rank
        if delivered == len(pods):
            best = configuration
            break
        expansions += 1
        # A newly picked pod can already be at its destination. A stationary
        # delivery pass is a legal action, too, and must precede more motion.
        settled = settle(positions, status)
        if settled != status:
            successors = [((positions, settled), 1, (None, None))]
        else:
            occupied = {node: positions.count(node) for node in positions}
            successors = []
            for u, node in enumerate(positions):
                for neighbor, weight in adjacency[node]:
                    if occupied.get(neighbor, 0) >= node_capacities[neighbor]:
                        continue
                    moved = positions[:u] + (neighbor,) + positions[u + 1:]
                    successors.append(((moved, settle(moved, status)), weight,
                                       (ids[u], neighbor)))
        for successor, weight, move in successors:
            candidate = elapsed + weight
            if candidate >= costs.get(successor, infinity):
                continue
            costs[successor] = candidate
            previous[successor] = configuration, move
            serial += 1
            heapq.heappush(queue, (candidate + heuristic(successor),
                                  -successor[1].count(-2), candidate, serial,
                                  successor))
    if best is None:
        remember(initial_key, None)
        return False, None
    path = []
    while best != initial:
        parent, action = previous[best]
        path.append((parent, action))
        best = parent
    # Cache the continuation as well as the first action. Replanning every
    # turn can choose a different retreat and oscillate forever.
    for configuration, action in reversed(path):
        remember(key(configuration), action)
    action = _cooperative_cache[initial_key]
    return True, action[1] if action[0] == drive_unit_id else None


def _delivery_order(layout, source, pods, now, first_arrivals=None):
    """Maximize discounted deliveries over a small set of station visits.

    The subset search is bounded; large loads use a greedy reward/travel rule.
    A partially reachable load still delivers every station it can reach.
    """
    rewards = {}
    for pod in pods:
        station = pod.destination_station
        rewards[station] = rewards.get(station, 0) + math.exp(
            -min(max(0, now - pod.entry_time), 10000) / 50.0)
    stations = tuple(sorted(rewards))

    def first_distance(station):
        return (first_arrivals.get(station, _INF) if first_arrivals is not None
                else layout.distance(source, station))

    if len(stations) > 7:
        order = []
        while stations:
            distances = {s: (first_distance(s) if not order else layout.distance(source, s))
                         for s in stations}
            reachable = [s for s in stations if distances[s] < _INF]
            if not reachable:
                break
            best = max(reachable, key=lambda s: (
                rewards[s] / (1 + distances[s]), -s))
            order.append(best)
            source = best
            stations = tuple(s for s in stations if s != best)
        return order

    @lru_cache(maxsize=None)
    def visit(node, mask):
        best_reward, best_order = -1.0, ()
        for index, station in enumerate(stations):
            if not mask & (1 << index):
                continue
            distance = (first_distance(station) if mask == (1 << len(stations)) - 1
                        else layout.distance(node, station))
            if distance == _INF:
                continue
            later, order = visit(station, mask ^ (1 << index))
            reward = math.exp(-distance / 50.0) * (rewards[station] + later)
            if reward > best_reward:
                best_reward, best_order = reward, (station,) + order
        return (max(0.0, best_reward), best_order)

    return visit(source, (1 << len(stations)) - 1)[1]


class _Traffic:
    def __init__(self, state, layout):
        self.state = state
        self.layout = layout
        self.units = {unit.id: unit for unit in state.drive_units}
        self.pods = {pod.id: pod for pod in state.active_pods}
        self.paths = {}
        self.reachable = {}
        ordered = sorted(state.drive_units, key=lambda unit: unit.id)
        if (ordered and len(ordered) <= 4 and len(layout.nodes) <= 10 and
                len(layout.nodes) ** len(ordered) <= 5000 and
                all(node[2] is not None for node in layout.nodes.values())):
            positions = tuple(self.location(unit) for unit in ordered)
            if all(position in layout.nodes for position in positions):
                self.reachable = dict(zip((unit.id for unit in ordered),
                                          _joint_reachability(layout, positions)))
        self.occupants = {node: [] for node in layout.nodes}
        self.edge_release = {}
        for unit in state.drive_units:
            node = unit.transit_destination if unit.in_transit else unit.current_node
            if node in self.occupants:
                self.occupants[node].append(unit)
        moving = [unit for unit in state.drive_units if unit.in_transit]
        for index, (source, destination, weight, capacity, both) in enumerate(layout.edges):
            if capacity is None:
                continue
            remaining = sorted(max(1, math.ceil(unit.transit_remaining_time))
                               for unit in moving
                               if (unit.current_node == source and
                                   unit.transit_destination == destination) or
                               (both and unit.current_node == destination and
                                unit.transit_destination == source))
            self.edge_release[index] = (remaining[len(remaining) - capacity]
                                        if len(remaining) >= capacity else 0)

    def location(self, unit):
        return unit.transit_destination if unit.in_transit else unit.current_node

    def load(self, unit):
        return [self.pods[pod] for pod in unit.carrying if pod in self.pods]

    def pickup_is_safe(self, unit, source, queue):
        """Check the real FIFO prefix; reservations cannot prevent auto-pickup."""
        batch = queue[:max(0, unit.capacity - len(unit.carrying))]
        if not batch:
            return True
        if unit.id in self.reachable and any(
                pod.destination_station not in self.reachable[unit.id] for pod in batch):
            return False
        destinations = [pod.destination_station for pod in batch]
        destinations.extend(pod.destination_station for pod in self.load(unit)
                            if self.layout.distance(self.location(unit), pod.destination_station) < _INF
                            and (unit.id not in self.reachable or
                                 pod.destination_station in self.reachable[unit.id]))
        if not self.layout.can_visit(source, destinations):
            return False
        for station in set(p.destination_station for p in batch):
            capacity = self.layout.nodes[station][2]
            if capacity is None or any(node != station for node in self.layout.adj[station]):
                continue
            # A dock without an exit is consumed permanently. Save its last
            # space for a larger available batch when that serves more pods.
            if capacity - len(self.occupants[station]) > 1:
                continue
            count = sum(p.destination_station == station for p in batch)
            demand = sum(p.carried_by is None and p.destination_station == station
                         for p in self.state.active_pods)
            if demand <= count:
                continue
            for other in self.units.values():
                if other.id == unit.id or self.layout.distance(self.location(other), source) == _INF:
                    continue
                prefix = queue[:max(0, other.capacity - len(other.carrying))]
                if sum(p.destination_station == station for p in prefix) <= count:
                    continue
                if other.id in self.reachable and any(
                        p.destination_station not in self.reachable[other.id] for p in prefix):
                    continue
                other_destinations = [p.destination_station for p in list(prefix) + self.load(other)]
                if self.layout.can_visit(source, other_destinations):
                    return False
        return True

    def valid(self, unit, neighbor):
        index = self.layout.adj.get(unit.current_node, {}).get(neighbor)
        if index is None or neighbor == unit.current_node:
            return False
        capacity = self.layout.nodes[neighbor][2]
        return (self.edge_release.get(index, 0) == 0 and
                (capacity is None or len(self.occupants[neighbor]) < capacity))

    def targets(self):
        """Allocate FIFO pickup batches, including robots already on an aisle."""
        layout = self.layout
        goals = {}
        for unit in self.units.values():
            deliverable = [pod for pod in self.load(unit)
                           if unit.id not in self.reachable or
                           pod.destination_station in self.reachable[unit.id]]
            first_arrivals = None
            if not unit.in_transit and len({p.destination_station for p in deliverable}) > 1:
                first_arrivals = {node: path[0] for node, path in self.routes(unit).items()}
            order = _delivery_order(layout, self.location(unit), deliverable,
                                    self.state.current_time_step, first_arrivals)
            if order:
                goals[unit.id] = order[0]
        waiting = {}
        for pod in self.state.active_pods:
            if pod.carried_by is None and pod.current_node in layout.nodes:
                waiting.setdefault(pod.current_node, []).append(pod)
        for pods in waiting.values():
            pods.sort(key=lambda p: (p.entry_time, p.id))
        pickup_queues = {source: tuple(pods) for source, pods in waiting.items()}
        available = {unit.id for unit in self.units.values()
                     if len(unit.carrying) < unit.capacity}

        bids = []
        versions = {source: 0 for source in waiting}

        def offer(uid, source):
            unit = self.units[uid]
            location = self.location(unit)
            travel = layout.distance(location, source)
            if travel == _INF or (uid in self.reachable and
                                  source not in self.reachable[uid]):
                return
            free = unit.capacity - len(unit.carrying)
            if not self.pickup_is_safe(unit, source, pickup_queues[source]):
                return
            batch = waiting[source][:free]
            feasible = [p for p in batch
                        if layout.distance(source, p.destination_station) < _INF and
                        (uid not in self.reachable or
                         p.destination_station in self.reachable[uid])]
            if not feasible:
                return
            if uid in goals:
                goal = goals[uid]
                if goal == location:
                    return
                direct = layout.distance(location, goal)
                detour = travel + layout.distance(source, goal) - direct
                # Pick up along a delivery route, without delaying a load
                # for a speculative, distant batch.
                # Collect reachable work before an irreversible delivery; once
                # inside a terminal branch this robot cannot return for it.
                if (detour > min(2.0, direct * 0.25) and
                        layout.distance(goal, source) < _INF):
                    return
            delay = math.ceil(unit.transit_remaining_time) if unit.in_transit else 0
            delivery = sum(layout.distance(source, p.destination_station)
                           for p in feasible) / len(feasible)
            cost = (1 + delay + travel + 0.6 * delivery) / len(feasible) ** 0.65
            heapq.heappush(bids, (cost, delay + travel, uid, source,
                                  len(batch), versions[source]))

        for uid in sorted(available):
            for source in waiting:
                offer(uid, source)
        while available and bids:
            _, _, uid, source, count, version = heapq.heappop(bids)
            if uid not in available or version != versions[source]:
                continue
            goals[uid] = source
            available.remove(uid)
            waiting[source] = waiting[source][count:]
            versions[source] += 1
            # Only this source's FIFO batch changed. Keep all other bids.
            if waiting[source]:
                for other_id in sorted(available):
                    offer(other_id, source)

        # Return spare robots to useful storage locations between pod arrivals.
        # Unlimited storage is a safe place to wait; docks are never parking.
        storage = [node for node, (_, kind, _) in layout.nodes.items() if kind == "storage"]
        for unit in sorted(self.units.values(), key=lambda u: u.id):
            if unit.id in goals:
                continue
            location = self.location(unit)
            candidates = []
            for node in storage:
                distance = layout.distance(location, node)
                if distance == _INF or (unit.id in self.reachable and
                                       node not in self.reachable[unit.id]):
                    continue
                # Idle repositioning must preserve the option to return, and
                # must obey the same automatic-pickup checks as assigned work.
                if layout.distance(node, location) == _INF or not self.pickup_is_safe(
                        unit, node, pickup_queues.get(node, ())):
                    continue
                # Automatic pickups make parking here unsafe if this robot
                # cannot physically reach the storage area's delivery docks.
                if unit.id in self.reachable and any(
                        kind == "station" and layout.distance(node, station) < _INF
                        and station not in self.reachable[unit.id]
                        for station, (_, kind, _) in layout.nodes.items()):
                    continue
                capacity = layout.nodes[node][2]
                others = sum(1 for other in self.units.values()
                             if other.id != unit.id and
                             (goals.get(other.id) == node or self.location(other) == node))
                if capacity is not None and others >= capacity:
                    continue
                candidates.append((distance + 2 * others, node))
            if candidates:
                goals[unit.id] = min(candidates)[1]
            else:
                # This also handles floors whose pickup nodes are not labelled
                # storage, and floors with unreachable storage components.
                parking = [(layout.distance(location, node), node)
                           for node, (_, kind, capacity) in layout.nodes.items()
                           if kind != "station" and capacity is None and
                           layout.distance(location, node) < _INF]
                if parking:
                    goals[unit.id] = min(parking)[1]
        return goals

    def node_release(self, node, unit):
        capacity = self.layout.nodes[node][2]
        occupants = [other for other in self.occupants[node] if other.id != unit.id]
        if capacity is None or len(occupants) < capacity:
            return 0
        releases = []
        for other in occupants:
            # Stationary robots can vacate on their next turn. Inbound robots
            # must arrive first. Replanning validates that prediction each tick.
            exits = self.layout.adj.get(node, {})
            if not any(neighbor != node for neighbor in exits):
                releases.append(_INF)
            else:
                releases.append(max(1, math.ceil(other.transit_remaining_time))
                                if other.in_transit else 1)
        releases.sort()
        return releases[len(releases) - capacity]

    def routes(self, unit):
        """Earliest arrivals, including aisle waits and inbound dock reservations."""
        if unit.id in self.paths:
            return self.paths[unit.id]
        start = unit.current_node
        distances = {start: 0}
        paths = {start: (0, None, 0)}
        queue = [(0, start, None, 0)]
        while queue:
            elapsed, node, first, first_departure = heapq.heappop(queue)
            if elapsed != distances[node]:
                continue
            for neighbor, index in self.layout.adj[node].items():
                if neighbor == node:
                    continue
                departure = max(elapsed, self.edge_release.get(index, 0),
                                self.node_release(neighbor, unit))
                arrival = departure + self.layout.edges[index][2]
                if arrival < distances.get(neighbor, _INF):
                    distances[neighbor] = arrival
                    hop = neighbor if first is None else first
                    leave = departure if first is None else first_departure
                    paths[neighbor] = (arrival, hop, leave)
                    heapq.heappush(queue, (arrival, neighbor, hop, leave))
        self.paths[unit.id] = paths
        return paths

    def next_step(self, unit, goal):
        """Compare taking a detour with waiting, and validate the first move."""
        _, first, departure = self.routes(unit).get(goal, (_INF, None, _INF))
        if first is not None and departure == 0 and self.valid(unit, first):
            return first
        return None

    def clear_blockage(self, unit, goals):
        """Yield into an available side aisle when another robot needs this slot."""
        node = unit.current_node
        kind, capacity = self.layout.nodes[node][1:]
        if capacity is None and kind != "station":
            return None
        requesters = []
        for other in self.units.values():
            if other.id == unit.id or other.in_transit or other.id not in goals:
                continue
            if self.layout.first_step(other.current_node, goals[other.id]) == node:
                requesters.append(other)
        # A loaded robot may productively wait at a dock for its next route.
        # Move it aside only for actual traffic; empty robots still clear docks.
        if not requesters and (kind != "station" or self.load(unit)):
            return None
        goal = goals.get(unit.id)
        candidates = []
        for neighbor, index in self.layout.adj[node].items():
            if not self.valid(unit, neighbor):
                continue
            # Never retreat into a one-way dead end with a deliverable load.
            onward = self.layout.distance(neighbor, goal) if goal is not None else 0
            if onward == _INF:
                continue
            neighbor_kind, neighbor_capacity = self.layout.nodes[neighbor][1:]
            penalty = 10 if neighbor_kind == "station" else 0
            penalty += 3 if neighbor_capacity is not None else 0
            # Another robot's goal is a last-resort retreat, not a forbidden
            # node: ruling it out can prevent a dock occupant from escaping.
            if any(neighbor == goals[other.id] and neighbor_capacity is not None
                   for other in requesters):
                penalty += 8
            if any(self.layout.distance(node, goals[other.id]) ==
                   self.layout.edges[index][2] +
                   self.layout.distance(neighbor, goals[other.id])
                   for other in requesters):
                penalty += 12
            candidates.append((penalty + self.layout.edges[index][2] + onward, neighbor))
        return min(candidates)[1] if candidates else None


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """Return a legal adjacent node, or wait for a pickup or a busy aisle.

    All decisions use the current snapshot, including earlier robots' moves.
    No pod schedules, test-case files, or state from earlier games are needed.
    """
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None
    nodes = tuple((node.id, node.node_type, node.capacity) for node in state.nodes)
    # The engine decrements once per tick, rounding EACH aisle up independently.
    edges = tuple((edge.from_node, edge.to_node, max(1, math.ceil(edge.weight)),
                   edge.capacity, edge.bidirectional) for edge in state.edges)
    layout = _layout(nodes, edges)
    if unit.current_node not in layout.nodes:
        return None
    traffic = _Traffic(state, layout)
    # A just-picked-up pod can already be at its destination. Let the next
    # automatic delivery pass run before leaving that node.
    if any(p.destination_station == unit.current_node for p in traffic.load(unit)):
        return None
    handled, move = _recover_if_stalled(state, layout, drive_unit_id)
    if handled:
        return move
    goals = traffic.targets()
    goal = goals.get(unit.id)
    if goal is not None:
        next_node = traffic.next_step(unit, goal)
        if next_node is not None:
            return next_node
    return traffic.clear_blockage(unit, goals)
