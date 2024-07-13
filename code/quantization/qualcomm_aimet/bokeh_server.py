from aimet_common.utils import start_bokeh_server_session
#from aimet_torch.visualize_serialized_data import VisualizeCompression

try:
    visualization_url, process = start_bokeh_server_session()
    print(visualization_url)

    # comp_ratios_file_path = './data/greedy_selection_comp_ratios_list.pkl'
    # eval_scores_path = './data/greedy_selection_eval_scores_dict.pkl'

    # # A user can visualize the eval scores dictionary and optimal compression ratios by executing the following code.
    # compression_visualizations = VisualizeCompression(visualization_url)
    # compression_visualizations.display_eval_scores(eval_scores_path)
    # compression_visualizations.display_comp_ratio_plot(comp_ratios_file_path)

    input("Press enter to quit Bokeh Server...")
    
finally:
    if process:
        process.terminate()
        process.join()