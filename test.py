elif sub_option == "多目标主动学习":
    colored_header(label="多目标主动学习",description=" ",color_name="violet-90")
    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed", accept_multiple_files=True)
    if len(file) != 2:
        st.error('Need two files, the first file is the train data and another is the visual data (test data)')
    else:
        df = pd.read_csv(file[0])
        # 检测缺失值
        check_string_NaN(df)

        colored_header(label="数据信息", description=" ",color_name="violet-70")
        nrow = st.slider("rows", 1, len(df)-1, 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)
        colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

        target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=2)
        
        col_feature, col_target = st.columns(2)
        # features
        features = df.iloc[:,:-target_num]
        # targets
        targets = df.iloc[:,-target_num:]
        with col_feature:    
            st.write(features.head())
        with col_target:   
            st.write(targets.head())

        visual_df = pd.read_csv(file[1])
        # 检测缺失值
        check_string_NaN(visual_df)
        
        col_feature, col_target = st.columns(2)
        # features
        vis_features = df.iloc[:,:-target_num]
        # targets
        vis_targets = df.iloc[:,-target_num:]

# =================== model ====================================
        reg = REGRESSOR(features,targets)

        colored_header(label="选择目标变量", description=" ", color_name="violet-70")
        target_selected_option = st.multiselect('target', list(reg.targets)[::-1], default=targets.columns.tolist())
        
        reg.targets = targets[target_selected_option]
        reg.Xtrain = features
        reg.Ytrain = targets
        reg.Xtest = vis_features
        reg.Ytest = vis_targets[target_selected_option]
        # st.write(reg.targets)
        if len(reg.targets.columns) != 2:
                st.error('Need two targets')
        colored_header(label="Multi-obj-opt", description=" ",color_name="violet-30")
        model_path = './models/multi-obj'

        template_alg = model_platform(model_path)
        inputs, col2 = template_alg.show()  

        if inputs['model'] == 'MOO':
            pareto_front = find_non_dominated_solutions(reg.targets.values, target_selected_option)
            pareto_front = pd.DataFrame(pareto_front, columns=target_selected_option)
            # st.write('pareto front of train data:', pareto_front)
            col1, col2 = st.columns([3, 1])
            with col1:
                with plt.style.context(['nature','no-latex']):
                    fig, ax = plt.subplots()
                    ax.plot(pareto_front[target_selected_option[0]], pareto_front[target_selected_option[1]], 'k--')
                    ax.scatter(reg.targets[target_selected_option[0]], reg.targets[target_selected_option[1]])
                    ax.set_xlabel(target_selected_option[0])
                    ax.set_ylabel(target_selected_option[1])
                    ax.set_title('Pareto front of visual space')
                    st.pyplot(fig)
            with col2:
                st.write(pareto_front)

            kernel = RBF(length_scale=1.0)
            reg.model = GaussianProcessRegressor(kernel=kernel)
            reg.model.fit(reg.Xtrain, reg.Ytrain)
            reg.Ypred, reg.Ystd = reg.model.predict(reg.Xtest, return_std=True)
            reg.Ypred = pd.DataFrame(reg.Ypred, columns=reg.Ytrain.columns.tolist())

            ref_point = [inputs['obj1 ref'], inputs['obj2 ref']]
            if inputs['method'] == 'HV':
                with st.container():
                    button_train = st.button('Opt', use_container_width=True)  
                if button_train:             
                    HV_values = []
                    for i in range(reg.Ypred.shape[0]):
                        i_Ypred = reg.Ypred.iloc[i]
                        Ytrain_i_Ypred = reg.Ytrain.append(i_Ypred)
                        i_pareto_front = find_non_dominated_solutions(Ytrain_i_Ypred.values, Ytrain_i_Ypred.columns.tolist())
                        i_HV_value = dominated_hypervolume(i_pareto_front, ref_point)
                        HV_values.append(i_HV_value)
                    
                    HV_values = pd.DataFrame(HV_values, columns=['HV values'])
                    HV_values.set_index(reg.Xtest.index, inplace=True)

                    max_idx = HV_values.nlargest(inputs['num'], 'HV values').index
                    recommend_point = reg.Xtest.loc[max_idx]
                    reg.Xtest = reg.Xtest.drop(max_idx)
                    st.write('The maximum value of HV:')
                    st.write(HV_values.loc[max_idx])
                    st.write('The recommended point is :')
                    st.write(recommend_point)
                    tmp_download_link = download_button(recommend_point, f'recommended samples.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
            elif inputs['method'] == 'EHVI':
                pass