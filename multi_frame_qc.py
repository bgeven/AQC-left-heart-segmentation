
from collections import defaultdict

def main_multi_frame_qc(all_files, views, cycle_information, multi_frame_qc_structural, multi_frame_qc_temporal):




    analysis_LV_2ch, analysis_LV_4ch, analysis_LA_2ch, analysis_LA_4ch = {}, {}, {}, {}

    for view in views:
        
        ED_points = cycle_information["ed_points_selected"][view]
        ES_point = cycle_information["ES_point_selected"][view]
          
        QC2_outliers_LV = multi_frame_qc_structural["flagged_frames_lv"][view]
        QC2_outliers_LA = multi_frame_qc_structural["flagged_frames_la"][view]
                
        dtw_LV = multi_frame_qc_temporal["dtw_lv"][view]
        dtw_LA = multi_frame_qc_temporal["dtw_la"][view]
        
        score_LV, score_LA = 0, 0
        
        nr_outliers_QC2_LV_cycle = count_values_between(QC2_outliers_LV, ED_points[0], ED_points[1]+1)
        nr_outliers_QC2_LA_cycle = count_values_between(QC2_outliers_LA, ED_points[0], ED_points[1]+1)
    
        # Outliers QC2
        if nr_outliers_QC2_LV_cycle > 1:
            score_LV += 1

        if nr_outliers_QC2_LA_cycle > 1:
            score_LA += 1
        
        if dtw_LV > 1:
            score_LV += 1

        if dtw_LA > 2:
            score_LA += 1
        
        # True if flagged, False if unflagged
        label_LV = True if score_LV >= 1 else False
        label_LA = True if score_LA >= 1 else False
        
        if image.endswith('a2ch'):
            analysis_LV_2ch[image] = label_LV
            analysis_LA_2ch[image] = label_LA
        elif image.endswith('a4ch'):
            analysis_LV_4ch[image] = label_LV
            analysis_LA_4ch[image] = label_LA
            
    analysis = {}
    analysis_all = {}
    analysis_LV_combined, analysis_LA_combined = {}, {}

    patients = sorted(set([i[:15] for i in all_files if i.startswith('cardiohance')]))
    for patient in patients:
        
        images_of_one_view_unsorted = [image for image in images if image.startswith(patient)]
        images_of_one_view = sorted(images_of_one_view_unsorted, key=lambda x: x[25:29])
        
        analysis_LV_combined[patient] = add_booleans(analysis_LV_2ch[images_of_one_view[0]], analysis_LV_4ch[images_of_one_view[1]])
        analysis_LA_combined[patient] = add_booleans(analysis_LA_2ch[images_of_one_view[0]], analysis_LA_4ch[images_of_one_view[1]])
        analysis_all[patient] = add_booleans(analysis_LV_combined[patient], analysis_LA_combined[patient])

    analysis['ALL'] = analysis_all
    analysis['LV'] = analysis_LV_combined
    analysis['LA'] = analysis_LA_combined