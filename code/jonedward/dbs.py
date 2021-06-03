import numpy as np


class Action:
    C = 1
    D = 0


class Memory:
    def __init__(self):
        self.alpha = 0.75
        self.promotion_threshold = 3
        self.violation_threshold = 4
        self.reject_threshold = 3
        self.tree_depth = 5
        self.history = []
        self.Rd = {
            (Action.C, Action.C): 1,
            (Action.C, Action.D): 1,
            (Action.D, Action.C): 0,
            (Action.D, Action.D): 0,
        }

        self.Rc = {}
        self.Pi = self.Rd
        self.violation_counts = {}

        self.v = 0

        self.history_by_cond = {
            (Action.C, Action.C): ([1], [1]),
            (Action.C, Action.D): ([1], [1]),
            (Action.D, Action.C): ([0], [1]),
            (Action.D, Action.D): ([0], [1]),
        }

    def should_promote(self, r_plus):
        if r_plus[1] == Action.C:
            opposite_action = 0
        else:
            opposite_action = 1
        k = 1
        count = 0
        while k < len(self.history_by_cond[r_plus[0]][0]) and not (
            self.history_by_cond[r_plus[0]][0][1:][-k] == opposite_action
            and self.history_by_cond[r_plus[0]][1][1:][-k] == 1
        ):
            if self.history_by_cond[r_plus[0]][1][1:][-k] == 1:
                count += 1
            k += 1
        if count >= self.promotion_threshold:
            return True
        return False

    def should_demote(self, r_minus):
        return self.violation_counts[r_minus[0]] >= self.violation_threshold

    def update_history_by_cond(self, opponent_history):
        two_moves_ago = (self.history[-2], opponent_history[-2])
        for outcome, GF in self.history_by_cond.items():
            g, f = GF
            if outcome == two_moves_ago:
                if opponent_history[-1] == Action.C:
                    g.append(1)
                else:
                    g.append(0)
                f.append(1)
            else:
                g.append(0)
                f.append(0)

    def compute_prob_rule(self, outcome):
        g = self.history_by_cond[outcome][0]
        f = self.history_by_cond[outcome][1]
        discounted_g = 0
        discounted_f = 0
        alpha_k = 1
        for g_i, f_i in zip(g[::-1], f[::-1]):
            discounted_g += alpha_k * g_i
            discounted_f += alpha_k * f_i
            alpha_k = self.alpha * alpha_k
        p_cond = discounted_g / discounted_f
        return p_cond


def strategy(history: np.ndarray, memory: Memory) -> (int, Memory):

    if memory is None:
        memory = Memory()
        return Action.C, memory

    lst_history: list = list(history)
    player_history = list(lst_history[0])
    opponent_history = list(lst_history[1])
    memory.history = player_history

    if len(memory.history) >= 2:
        memory.update_history_by_cond(opponent_history)
        two_moves_ago = (memory.history[-2], opponent_history[-2])
        r_plus = (two_moves_ago, opponent_history[-1])
        r_minus = (two_moves_ago, ({Action.C, Action.D} - {opponent_history[-1]}).pop())

        if r_plus[0] not in memory.Rc.keys():
            if memory.should_promote(r_plus):
                memory.Rc[r_plus[0]] = r_plus[1]
                memory.violation_counts[r_plus[0]] = 0
                memory.violation_counts[r_plus[0]] = 0

        if r_plus[0] in memory.Rc.keys():
            to_check = Action.C if memory.Rc[r_plus[0]] == 1 else Action.D
            if r_plus[1] == to_check:
                memory.violation_counts[r_plus[0]] = 0
            elif r_minus[1] == to_check:
                memory.violation_counts[r_plus[0]] += 1
                if memory.should_demote(r_minus):
                    memory.Rd.update(memory.Rc)
                    memory.Rc.clear()
                    memory.violation_counts.clear()
                    memory.v = 0

        r_plus_in_rc = (
            r_plus[0] in memory.Rc.keys() and memory.Rc[r_plus[0]] == r_plus[1]
        )

        r_minus_in_rd = (
            r_minus[0] in memory.Rd.keys() and memory.Rd[r_minus[0]] == r_minus[1]
        )

        if r_minus_in_rd:
            memory.v += 1

        if (memory.v > memory.reject_threshold) or (r_plus_in_rc and r_minus_in_rd):
            memory.Rd.clear()
            memory.v = 0

        rp = {}
        all_cond = [
            (Action.C, Action.C),
            (Action.C, Action.D),
            (Action.D, Action.C),
            (Action.D, Action.D),
        ]
        for outcome in all_cond:
            if (outcome not in memory.Rc.keys()) and (outcome not in memory.Rd.keys()):
                rp[outcome] = memory.compute_prob_rule(outcome)

        memory.Pi = {}
        memory.Pi.update(memory.Rc)
        memory.Pi.update(memory.Rd)
        memory.Pi.update(rp)

    return (
        move_gen(
            (memory.history[-1], opponent_history[-1]),
            memory.Pi,
            depth_search_tree=memory.tree_depth,
        ),
        memory,
    )


class Node(object):

    # abstract method
    def get_siblings(self, policy):
        raise NotImplementedError("subclasses must override get_siblings()!")

    # abstract method
    def is_stochastic(self):
        raise NotImplementedError("subclasses must override is_stochastic()!")


class StochasticNode(Node):
    def __init__(self, own_action, pc, depth):
        self.pC = pc
        self.depth = depth
        self.own_action = own_action

    def get_siblings(self, policy=None):
        opponent_c_choice = DeterministicNode(self.own_action, Action.C, self.depth + 1)
        opponent_d_choice = DeterministicNode(self.own_action, Action.D, self.depth + 1)
        return opponent_c_choice, opponent_d_choice

    def is_stochastic(self):
        return True


class DeterministicNode(Node):
    def __init__(self, action1, action2, depth):
        self.action1 = action1
        self.action2 = action2
        self.depth = depth

    def get_siblings(self, policy):
        c_choice = StochasticNode(
            Action.C, policy[(self.action1, self.action2)], self.depth
        )
        d_choice = StochasticNode(
            Action.D, policy[(self.action1, self.action2)], self.depth
        )
        return c_choice, d_choice

    def is_stochastic(self):
        return False

    def get_value(self):
        values = {
            (Action.C, Action.C): 3,
            (Action.C, Action.D): 0,
            (Action.D, Action.C): 5,
            (Action.D, Action.D): 1,
        }
        return values[(self.action1, self.action2)]


def minimax_tree_search(begin_node, policy, max_depth):
    if begin_node.is_stochastic():
        siblings = begin_node.get_siblings()
        node_value = begin_node.pC * minimax_tree_search(
            siblings[0], policy, max_depth
        ) + (1 - begin_node.pC) * minimax_tree_search(siblings[1], policy, max_depth)
        return node_value
    else:
        if begin_node.depth == max_depth:
            return begin_node.get_value()
        elif begin_node.depth == 0:
            siblings = begin_node.get_siblings(policy)
            return (
                minimax_tree_search(siblings[0], policy, max_depth)
                + begin_node.get_value(),
                minimax_tree_search(siblings[1], policy, max_depth)
                + begin_node.get_value(),
            )
        elif begin_node.depth < max_depth:
            siblings = begin_node.get_siblings(policy)
            a = minimax_tree_search(siblings[0], policy, max_depth)
            b = minimax_tree_search(siblings[1], policy, max_depth)
            node_value = max(a, b) + begin_node.get_value()
            return node_value


def move_gen(outcome, policy, depth_search_tree=5):
    current_node = DeterministicNode(outcome[0], outcome[1], depth=0)
    values_of_choices = minimax_tree_search(current_node, policy, depth_search_tree)
    actions_tuple = (Action.C, Action.D)
    return actions_tuple[values_of_choices.index(max(values_of_choices))]
