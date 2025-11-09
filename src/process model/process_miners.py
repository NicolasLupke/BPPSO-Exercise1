import pm4py




if __name__ == "__main__":
    log = pm4py.read_xes("Dataset\BPI Challenge 2017.xes")
    alpha_miner = pm4py.discover_process_tree_alpha(log)
    inductive_miner = pm4py.discover_process_tree_inductive(log)
    heuristics_miner = pm4py.discover_heuristics_net(log)
    dfg_miner = pm4py.discover_dfg(log)
    process_tree = pm4py.discover_process_tree_inductive(log)
    bpmn_model = pm4py.convert_to_bpmn(process_tree)
    pm4py.view_bpmn(bpmn_model)