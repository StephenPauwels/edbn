"""
    Implementation of anomaly detection algorithm in multidimensional sequential data:
       [1] Böhmer, Kristof, and Stefanie Rinderle-Ma. "Multi-perspective anomaly detection in business process execution events."
             OTM Confederated International Conferences" On the Move to Meaningful Internet Systems". Springer, Cham, 2016.
"""
import pandas as pd

CASE_ATTR = "case_id"
ACTIVITY_ATTR = "name"
RESOURCE_ATTR = "user"
WEEKDAY_ATTR = "day"

class LikelihoodModel:

    def __init__(self, data, act_idx=1, res_idx=2, wk_idx=3):
        self.dict_to_id = dict()
        self.dict_to_value = dict()

        self.cache_dict = dict()

        self.dependencies = dict()

        self.dict_evntTypLkly = dict()
        self.dict_minLike = dict()

        self.data = data
        self.graph = None

        self.act_idx = act_idx
        self.res_idx = res_idx
        self.wk_idx = wk_idx

    def basicLikelihoodGraph(self):
        V = set()
        self.dict_to_value[0] = "START"
        V.add(0)
        self.dict_to_value[1] = "END"
        V.add(1)
        D = set()
        self.dependencies[0] = []
        self.dependencies[1] = []

        grouped_logs = self.data.data.groupby(CASE_ATTR) # Group log file according to Cases
        i = 0
        activity_mapping = {}
        for name, group in grouped_logs: # Iterate over all groupes
            print("Case", i, "/", len(grouped_logs))
            i += 1
            a_lst = 0
            for row in pd.DataFrame(group).itertuples(index = False): # Iterate over rows in group
                activity = getattr(row, self.data.activity)
                if activity not in activity_mapping:
                    node_id = len(self.dict_to_value)
                    self.dependencies[node_id] = []
                    self.dict_to_value[node_id] = activity
                    activity_mapping[activity] = node_id
                else:
                    node_id = activity_mapping[activity]

                V.add(node_id)
                if node_id not in self.dependencies[a_lst]:
                    D.add((a_lst, node_id, self.likeA(a_lst, node_id)))
                    self.dependencies[a_lst].append(node_id)
                a_lst = node_id
            D.add((a_lst, 1, 1))
            self.dependencies[a_lst].append((1,1))
        self.graph = (V, D)


    def likeA(self, a_s, a_e):
        a_s = self.dict_to_value[a_s]
        a_e = self.dict_to_value[a_e]
        if a_s == "START":
            return 1

        tc = 0
        ec = 0

        log = self.data.data
        filtered = log.loc[log[self.data.activity] == a_s]
        tc = len(filtered)
        for idx in filtered.index:
            if idx + 1 in log.index and log.at[idx + 1, CASE_ATTR] == log.at[idx, CASE_ATTR] and log.at[idx + 1, ACTIVITY_ATTR] == a_e:
                ec += 1

        return ec / tc


    def extendLikelihoodGraph(self):
        F = set()
        V = self.graph[0]
        D = self.graph[1]
        v_cnt = 0

        logs = self.data.data

        for v in V:
            print("Variable", v_cnt, "/", len(V))
            v_cnt += 1
            if v == 0 or v == 1: # 0 and 1 are predefined as START and END
                continue
            V_next = {x for x in D if x[0] == v}
            D = D.difference(V_next)
            self.dependencies[v] = [x for x in self.dependencies[v] if x == 1] # Reset dependencies for v, only keep dependency to END
            activity_filtered = logs.loc[logs[ACTIVITY_ATTR] == self.dict_to_value[v]]
            E_r = set(x[self.res_idx] for x in activity_filtered.itertuples(index=False)) # SELECT resources with activity == v
            for r in E_r:
                r_node_id = len(self.dict_to_value)
                self.dict_to_value[r_node_id] = r
                self.dict_to_id[r] = r_node_id
                F.add(r_node_id)
                like_g = self.likeG(activity_filtered, v, None, r, None, "resource")
                D.add((v, r_node_id, like_g))
                self.dependencies[v].append((r_node_id, like_g)) # Add dependency from v to r_node_id (from ACTIVITY -> RESOURCE)
                self.dependencies[r_node_id] = [] # Init new dependency for resource node
                activity_resource_filtered = activity_filtered.loc[activity_filtered[RESOURCE_ATTR] == r]
                E_wd = set(x[self.wk_idx] for x in activity_resource_filtered.itertuples(index=False))
                for wd in E_wd:
                    wd_node_id = len(self.dict_to_value)
                    self.dict_to_value[len(self.dict_to_value)] = wd
                    F.add(wd_node_id)
                    like_g = self.likeG(activity_resource_filtered, v, None, r, wd, "weekday")
                    D.add((r_node_id, wd_node_id, like_g))
                    self.dependencies[r_node_id].append((wd_node_id, like_g)) # Add dependency from r_node_id to wd_node_id (from RESOURCE -> WEEKDAY)
                    self.dependencies[wd_node_id] = [] # Init new dependency for resource node
                    for v_next in V_next:
                        likely = self.likeG((logs, activity_resource_filtered), v, v_next[1], r, wd, "final")
                        if likely > 0:
                            D.add((wd_node_id, v_next[1], likely))
                            self.dependencies[wd_node_id].append((v_next[1], likely))
        self.graph = (V.union(F), D)


    def likeG(self, logs, a_s, a_e, r, wd, type):
        cache_tuple = (a_s, a_e, r, wd, type)
        if cache_tuple in self.cache_dict:
            return self.cache_dict[cache_tuple]

        tc = 0
        ec = 0

        if type == "resource":
            tc = len(logs)
            ec = len(logs.loc[logs[RESOURCE_ATTR] == r])
        elif type == "weekday":
            tc = len(logs)
            ec = len(logs.loc[logs[WEEKDAY_ATTR] == wd])
        elif type == "final":
            log = logs[0]
            filtered = logs[1]
            filtered = filtered.loc[filtered[WEEKDAY_ATTR] == wd]
            tc = len(filtered)
            for idx in filtered.index:
                if idx + 1 in log.index and log.at[idx + 1, CASE_ATTR] == log.at[idx, CASE_ATTR] and \
                        (log.at[idx + 1, ACTIVITY_ATTR] == self.dict_to_value[a_e] or log.at[idx + 1, ACTIVITY_ATTR] == self.dict_to_value[1]):
                    ec += 1

        if tc == 0 or ec == 0:
            return 0

        self.cache_dict[cache_tuple] = ec / tc

        return ec / tc

    def mapEvents(self, lst_v, lst_va, f, lst_l, punAct, punOth):
        D = {x for x in self.graph[1] if x[0] == lst_v}
        fnd = False
        likly = 0
        for d in D:
 #           print(self.dict_to_value[d[1]] == f, d[1], self.dict_to_value[d[1]], f, type(self.dict_to_value[d[1]]), type(f))
            if self.dict_to_value[d[1]] == f: # f is a successor of the last successfully mapped event lst_v then
                lst_v = d[1]
                likly = d[2]
                fnd = True
                break
        if not fnd: # if the event f was not recorded in L then
            pun = punAct if self.isActivity(f) else punOth
            if lst_va is not None:
                tmp = self.evntTypLkly(f, lst_va)
                f_avglkli = [x[1] for x in tmp if x[0] == f]
                if len(f_avglkli) != 0:
                    likly = f_avglkli[0] * pun
                else:
                    likyhds = [x[1] for x in tmp]
                    gLkli = 1 - self.gini(sorted(list(likyhds)), len(likyhds))
                    cLkly = 1 - self.classLkly(f, lst_va, lst_v)
                    likly = gLkli * cLkly * pun
            else:
                likly = lst_l * pun
        if self.isActivity(f):
            matchingActivities = [x for x in self.graph[0] if self.dict_to_value[x] == f]
            if len(matchingActivities) > 0:
                lst_v = matchingActivities[0]
                lst_va = lst_v
            else:
                lst_va = None
        return lst_v, lst_va, likly * lst_l

    def minLike(self, a, s_max):
        if (a, s_max) in self.dict_minLike:
            return self.dict_minLike[(a, s_max)]

        logs = self.data.data

        min = 1
        grouped = logs.groupby(CASE_ATTR)
        for name, group in grouped:
            s_c = 0
            found_a = False
            l_c = 1
            for e_idx in group.index:
                e = list(group.loc[e_idx])
                s_c += 1
                # Determine l_a2r (ACTIVITY -> RESOURCE)
                id = -1 # Find ID for ACTIVITY NODE
                for key in self.dependencies.keys():
                    if self.dict_to_value[key] == e[self.act_idx]:
                        id = key
                        break

                try:
                    l_a2r = [x for x in self.dependencies[id] if self.dict_to_value[x[0]] == e[self.res_idx]][0] # (resource_id, likely)
                except IndexError:
                    print("Error:", e)
                    print(id, self.dependencies[id])
                # Determine l_r2wd (RESOURCE -> WEEKDAY)
                l_r2wd = [x for x in self.dependencies[l_a2r[0]] if self.dict_to_value[x[0]] == e[self.wk_idx]][0] # (weekday_id, likely)
                # Determine l_wd2a (WEEKDAY -> ACTIVITY
                if e_idx + 1 in group.index:
                    l_wd2a = [x for x in self.dependencies[l_r2wd[0]] if self.dict_to_value[x[0]] == group.at[e_idx + 1, ACTIVITY_ATTR]][0] # (activity, likely) | TODO: Possible to improve?
                else:
                    l_wd2a = (0,1)
                l_c = l_c * l_a2r[1] * l_r2wd[1] * l_wd2a[1]
                if a in self.dict_to_value and e[0] == self.dict_to_value[a] and s_c == s_max:
                    found_a = True
                    break
                elif s_c > s_max:
                    break
            if found_a and l_c < min:
                min = l_c
            elif not found_a:
                min = 0
                self.dict_minLike[(a, s_max)] = min
        return min

    def isActivity(self, node):
        if node.startswith("Activity") or (not node.startswith("r_") and not node.startswith("wd_")):
            return True
        return False
    # Addapted for Nolle's format
#        return node.startswith("Activity") #TODO: changed from a_ to Activity

    def isRes(self, node):
        return node.startswith("r_")

    def isWeekday(self, node):
        return node.startswith("wd_")

    def evntTypLkly(self, f, lst_va):
        if (f, lst_va) in self.dict_evntTypLkly:
            return self.dict_evntTypLkly[(f, lst_va)]

        E = [(d,d[2]) for d in self.graph[1] if d[0] == lst_va]
        E_extend = E.extend

        fnd_fs = set()
        kwn_e = set()
        kwn_e_add = kwn_e.add
        i = 0
        while i < len(E):
            e = E[i]
            i += 1
            kwn_e_add(e[0])
            if self.getType(self.dict_to_value[e[0][1]]) == self.getType(f):
                fnd_f = [x for x in fnd_fs if x[0] == self.dict_to_value[e[0][1]]]
                if len(fnd_f) == 0:
                    fnd_fs.add((self.dict_to_value[e[0][1]], e[1]))
                else:
                    fnd_fs = fnd_fs.difference(fnd_f)
                    fnd_fs.add((fnd_f[0][0], fnd_f[0][1] + e[1]))
            else:
                E_extend([(x, e[1] * x[2]) for x in self.graph[1] if x[0] == e[0][1] and x not in kwn_e])
        self.dict_evntTypLkly[(f, lst_va)] = fnd_fs
        return fnd_fs

    def getType(self, node):
        if node.startswith("Activity") or (not node.startswith("r_") and not node.startswith("wd_")): #if node.startswith("Activity"):
            return ACTIVITY_ATTR
        elif node.startswith("r_"):
            return RESOURCE_ATTR
        elif node.startswith("wd_"):
            return WEEKDAY_ATTR

    def classLkly(self, f, lst_a, lst_v):
        logs = self.data.data

        if self.isRes(f):
            tc = len(logs[RESOURCE_ATTR].unique()) # total amount of resources
            ec = len(logs.loc[logs[ACTIVITY_ATTR] == self.dict_to_value[lst_a]][RESOURCE_ATTR].unique()) # total amount of resources with given activity
        elif self.isWeekday(f):
            tc = len(logs[WEEKDAY_ATTR].unique()) # total amount of weekdays
            if self.isRes(self.dict_to_value[lst_v]):
                ec = len(logs.loc[(logs[ACTIVITY_ATTR] == self.dict_to_value[lst_a]) & (logs[RESOURCE_ATTR] == self.dict_to_value[lst_v])]) # total amount of weekdays with given activity and resource
            else:
                ec = len(logs.loc[logs[ACTIVITY_ATTR] == self.dict_to_value[lst_a]][WEEKDAY_ATTR].unique())
        else:
            filtered = logs.loc[logs[ACTIVITY_ATTR] == self.dict_to_value[lst_a]]
            tc = len({logs.at[idx + 1, ACTIVITY_ATTR] for idx in filtered.index if idx + 1 in logs.index and logs.at[idx, CASE_ATTR] == logs.at[idx + 1, CASE_ATTR]})

            if self.isRes(self.dict_to_value[lst_v]):
                filtered = filtered.loc[filtered[RESOURCE_ATTR] == self.dict_to_value[lst_v]]
                ec = len({logs.at[idx + 1, ACTIVITY_ATTR] for idx in filtered.index if idx + 1 in logs.index and logs.at[idx, CASE_ATTR] == logs.at[idx + 1, CASE_ATTR]})
            else:
                filtered = filtered.loc[filtered[WEEKDAY_ATTR] == self.dict_to_value[lst_v]]
                ec = len({logs.at[idx + 1, ACTIVITY_ATTR] for idx in filtered.index if idx + 1 in logs.index and logs.at[idx, CASE_ATTR] == logs.at[idx + 1, CASE_ATTR]})
        try:
            max = 0.5
            min = 1 / (tc + 1)
            rawLkly = ec / tc
            if rawLkly > max:
                rawLkly = 1 - rawLkly
            return (rawLkly - min) / (max - min)
        except ZeroDivisionError:
            return 0


    def gini(self, x,n):
        tmp = 0
        for i in range(0, n):
            tmp += (i+1) * x[i]
        if n == 0:
            return 0
        return (2 * tmp) / (n * sum(x)) - (n+1) / n


    def ongoingLikelihoodDiff(self, trace):
        lst_v = 0
        lst_va = None
        lst_l = 1
        punAct = 0.9
        punOth = 0.95
        min_prob = 1

        i = 0
        for idx, row in trace.iterrows():
            i += 1
            for attr in [ACTIVITY_ATTR, RESOURCE_ATTR, WEEKDAY_ATTR]:
                f = row[attr]
                lst_v, lst_va, lst_l = self.mapEvents(lst_v, lst_va, f, lst_l, punAct, punOth)
            #min_lik = self.minLike(lst_va, i)
            #prob = lst_l - min_lik
            prob = lst_l
            if prob < min_prob:
                min_prob = prob
        return min_prob


    def test_trace(self, trace):
        """
        Return score for a trace given the graph and log

        :param trace: Current trace to score
        :return:
        """
        print("Testing")
        return self.ongoingLikelihoodDiff(trace)

