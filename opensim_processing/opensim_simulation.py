import os
import opensim as osim
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat
# from visualization.matplotly_plot import plot_imu_osimimu
from utils.read_write import read_xsens_imu, read_opensim_sto_mot
from utils import filter


class SimulateData:
    def __init__(self, osim_model, KIHA_IKSetUp, KIHA_AnalyzeSetUp):
        self.model = osim_model
        self.ik_setup_file = KIHA_IKSetUp
        self.analyze_setup_file = KIHA_AnalyzeSetUp
        self.analyzeTool = osim.AnalyzeTool(self.analyze_setup_file)

    def change_joint_range(self, ):
        return

    def change_joint_coordinate(self, ):
        return

    def run_ik(self, trc_file, ik_result_folder):
        model = osim.Model(self.model)
        model.initSystem()
        markerData = osim.MarkerData(trc_file)
        initial_time = osim.markerData.getStartFrameTime()
        final_time = osim.markerData.getLastFrameTime()

        ikTool = osim.InverseKinematicsTool(self.ik_setup_file)
        # ikTool.setName(trial_number)
        ikTool.setModel(model)
        ikTool.setMarkerDataFileName(trc_file)
        ikTool.setStartTime(initial_time)
        ikTool.setEndTime(final_time)
        ikTool.setOutputMotionFileName(ik_result_folder)
        ikTool.setResultsDir(ik_result_folder)
        ikTool.run()

    def run_analyze(self, ik_file, analyze_result_folder):
        '''
        :param ik_file: IK
        :param analyze_result_folder:
        :return: write analyze files in the given folder
        :Note: self analyze tool should be within the function when we run augmentation and should be delete after each iteration
        . we can also move the self.analyzeTool up when we set the class
        '''
        motCoordsData = osim.Storage(ik_file)
        initial_time = motCoordsData.getFirstTime()
        final_time = motCoordsData.getLastTime()
        start = [i for i, c in enumerate(ik_file) if c=='/'][-1]+1
        name = ik_file[start:][:-4]
        self.analyzeTool.setName(name)
        self.analyzeTool.setModelFilename(self.model)
        self.analyzeTool.setResultsDir(analyze_result_folder)
        self.analyzeTool.setCoordinatesFileName(ik_file)
        self.analyzeTool.setInitialTime(initial_time)
        self.analyzeTool.setFinalTime(final_time)
        self.analyzeTool.setLowpassCutoffFrequency(6)
        analysis_set = self.analyzeTool.getAnalysisSet()
        nAnalysis = analysis_set.getSize()
        for k in range(nAnalysis):
            analysis = analysis_set.get(k)
            analysis.setStartTime(initial_time)
            analysis.setEndTime(final_time)
        self.analyzeTool.run()

    def run_simulating_imu(self, analyze_result_folder, trial_num):
        model_osim = osim.Model(self.model)
        imus_coordinate = self.get_imu_coordinates(model_osim)
        del model_osim
        imu_marker_list = ['PelvisIMU', 'RUpLegIMU', 'RLoLegIMU', 'RFootIMU',
                           'LUpLegIMU', 'LLoLegIMU', 'LFootIMU']
        s_index = analyze_result_folder.find('/S')
        subject_num = analyze_result_folder[s_index+1:s_index+4]
        r_mat_file_name = 'F:/dataset/opensim_imu_augmentation/Models/r_mat_freeacc/' + subject_num + '_r_mat_freeacc.p'
        if os.path.exists(r_mat_file_name):
            with open(r_mat_file_name, 'rb') as f:
                r_mat_freeacc = pickle.load(f)
        else:
            r_mat_freeacc = {}
        osim_imu = {}
        for imu, imu_name in enumerate(imu_marker_list):
            if imu_name == 'PelvisIMU':
                opensim_body_name = 'pelvis'
                euler_trans_num = 0
            elif imu_name == 'RUpLegIMU':
                opensim_body_name = 'femur_r'
                euler_trans_num = 1
            elif imu_name == 'RLoLegIMU':
                opensim_body_name = 'tibia_r'
                euler_trans_num = 2
            elif imu_name == 'RFootIMU':
                opensim_body_name = 'calcn_r'
                euler_trans_num = 3
            elif imu_name == 'LUpLegIMU':
                opensim_body_name = 'femur_l'
                euler_trans_num = 4
            elif imu_name == 'LLoLegIMU':
                opensim_body_name = 'tibia_l'
                euler_trans_num = 5
            elif imu_name == 'LFootIMU':
                opensim_body_name = 'calcn_l'
                euler_trans_num = 6
            # import body kinematic, opensim output
            bodyVel_l = read_opensim_sto_mot(analyze_result_folder + '/' + trial_num +'_BodyKinematics_vel_bodyLocal.sto')
            # import point kinematic, opensim output
            PointVel_g = read_opensim_sto_mot(analyze_result_folder + '/' + trial_num + '_PointKinematics_' + imu_name + '_' + imu_name + '_vel.sto')

            gyr_l_opensim_body = bodyVel_l[[opensim_body_name+'_Ox', opensim_body_name+'_Oy', opensim_body_name+'_Oz']].values
            vel_g_opensim_point = PointVel_g[['state_0', 'state_1', 'state_2']].values
            freeacc_g_opensim_point = np.diff(vel_g_opensim_point, axis=0) / 0.01

            # free acc r mat
            try:
                r_mat = r_mat_freeacc[imu_name]
            except:
                print('fee acc is empty, should run_verfication.py to create rotation matrix for free acc for each subject')
                r_mat = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0]])
            # multiply rotation matrix with opensim freeAcc global to get simulated freeAcc
            freeacc_osimimu = self.simulate_freeacc(freeacc_g_opensim_point, r_mat)
            # gyr r mat
            R_euler_Mocap = np.flip(imus_coordinate[euler_trans_num]['r_eulor'])
            r_mat = euler2mat(R_euler_Mocap[0], R_euler_Mocap[1], R_euler_Mocap[2], axes='szyx')
            gyr_osimimu = self.simulate_gyro(gyr_l_opensim_body, r_mat)

            data = np.concatenate([freeacc_osimimu, gyr_osimimu[0:len(freeacc_osimimu), :]], axis=1)
            columns = ['freeacc_x', 'freeacc_y', 'freeacc_z', 'gyr_x', 'gyr_y', 'gyr_z']
            osim_imu_df = pd.DataFrame(data=data, columns = columns)
            osim_imu[imu_name] = osim_imu_df

        return osim_imu

    def verify_simulated_imu(self, analyze_result_folder, trial_num, xsens_imu_folder):
        model_osim = osim.Model(self.model)
        imus_coordinate = self.get_imu_coordinates(model_osim)
        imu_marker_list = ['PelvisIMU', 'RUpLegIMU', 'RLoLegIMU', 'RFootIMU',
            'LUpLegIMU', 'LLoLegIMU', 'LFootIMU']
        subject_num = analyze_result_folder[-12:][0:3]
        r_mat_file_name = 'F:/dataset/opensim_imu_augmentation/Models/r_mat_freeacc/' + subject_num + '_r_mat_freeacc.p'
        if os.path.exists(r_mat_file_name):
            with open(r_mat_file_name, 'rb') as f:
                r_mat_freeacc = pickle.load(f)
        else:
            r_mat_freeacc = {}
        for imu, imu_name in enumerate(imu_marker_list):
            if imu_name == 'PelvisIMU':
                sensor_name = 'PELVIS'
                opensim_body_name = 'pelvis'
                euler_trans_num = 0
            elif imu_name == 'RUpLegIMU':
                sensor_name = 'Right uLEG'
                opensim_body_name = 'femur_r'
                euler_trans_num = 1
            elif imu_name == 'RLoLegIMU':
                if subject_num == 'S39':
                    sensor_name = 'Left lLeg'
                else:
                    sensor_name = 'Right lLeg'
                opensim_body_name = 'tibia_r'
                euler_trans_num = 2
            elif imu_name == 'RFootIMU':
                sensor_name = 'Right Foot'
                opensim_body_name = 'calcn_r'
                euler_trans_num = 3
            elif imu_name == 'LUpLegIMU':
                sensor_name = 'Left uLEG'
                opensim_body_name = 'femur_l'
                euler_trans_num = 4
            elif imu_name == 'LLoLegIMU':
                if subject_num == 'S39':
                    sensor_name = 'Right lLeg'
                else:
                    sensor_name = 'Left lLeg'
                opensim_body_name = 'tibia_l'
                euler_trans_num = 5
            elif imu_name == 'LFootIMU':
                sensor_name = 'Left Foot'
                opensim_body_name = 'calcn_l'
                euler_trans_num = 6
            # import xsense sensor
            # load xsens imu data imuosim_result_folder
            imu_data = read_xsens_imu(xsens_imu_folder + '/' + sensor_name +'.txt')
            acc_g = imu_data[['acc_g_x', 'acc_g_y', 'acc_g_z']].values
            acc_s = imu_data[['acc_s_x', 'acc_s_y', 'acc_s_z']].values
            gyr_g = imu_data[['gyr_g_x', 'gyr_g_y', 'gyr_g_z']].values
            gyr_s = imu_data[['gyr_s_x', 'gyr_s_y', 'gyr_s_z']].values
            freeacc_g = imu_data[['freeacc_g_x', 'freeacc_g_y', 'freeacc_g_z']].values
            freeacc_s = imu_data[['freeacc_s_x', 'freeacc_s_y', 'freeacc_s_z']].values
            # Filter IMU data
            acc_g_imu = filter.butter_lowpass_filter(acc_g, lowcut=6, fs=100, order=2)
            acc_g_imu[:, 2] = acc_g_imu[:, 2]-9.81
            acc_s_imu = filter.butter_lowpass_filter(acc_s, lowcut=6, fs=100, order=2)
            freeacc_g_imu = filter.butter_lowpass_filter(freeacc_g, lowcut=6, fs=100, order=2)
            freeacc_s_imu = filter.butter_lowpass_filter(freeacc_s, lowcut=6, fs=100, order=2)
            gyr_g_imu = filter.butter_lowpass_filter(gyr_g, lowcut=6, fs=100, order=2)
            gyr_s_imu = filter.butter_lowpass_filter(gyr_s, lowcut=6, fs=100, order=2)

            # import body kinematic, opensim output
            read_opensim_sto_mot(analyze_result_folder + '/' + trial_num + '_BodyKinematics_vel_global.sto')
            bodyVel_g = read_opensim_sto_mot(analyze_result_folder + '/' + trial_num + '_BodyKinematics_vel_global.sto')
            bodyVel_l = read_opensim_sto_mot(analyze_result_folder + '/' + trial_num + '_BodyKinematics_vel_bodyLocal.sto')
            bodyAcc_g = read_opensim_sto_mot(analyze_result_folder + '/' + trial_num + '_BodyKinematics_acc_global.sto')
            bodyAcc_l = read_opensim_sto_mot(analyze_result_folder + '/' + trial_num + '_BodyKinematics_acc_bodyLocal.sto')
            # import point kinematic, opensim output
            PointAcc_g = read_opensim_sto_mot(analyze_result_folder + '/' + trial_num + '_PointKinematics_'+imu_name +'_' + imu_name + '_acc.sto')
            PointVel_g = read_opensim_sto_mot(analyze_result_folder + '/' + trial_num + '_PointKinematics_'+imu_name +'_' + imu_name + '_vel.sto')
            # Body Based Opensim Gyro and Acceleration
            gyr_g_opensim_body = bodyVel_g[[opensim_body_name+'_Ox', opensim_body_name+'_Oy', opensim_body_name+'_Oz']].values
            gyr_l_opensim_body = bodyVel_l[[opensim_body_name+'_Ox', opensim_body_name+'_Oy', opensim_body_name+'_Oz']].values
            # Point-based Opensim Acceleration
            acc_g_opensim_point = PointAcc_g[['state_0', 'state_1', 'state_2']].values
            vel_g_opensim_point = PointVel_g[['state_0', 'state_1', 'state_2']].values
            freeacc_g_opensim_point = np.diff(vel_g_opensim_point, axis=0) / 0.01
            # simulate FreeACC
            # get rotation matrix between opensim freeAcc global and IMU freeAcc global
            if not len(r_mat_freeacc) == 7:
                r = np.dot(freeacc_s_imu[0:len(freeacc_g_opensim_point), :].transpose(), freeacc_g_opensim_point)
                r_mat = quat2mat(mat2quat(r))
                r_mat_freeacc[imu_name] = r_mat
                if imu == 6:
                    with open(r_mat_file_name, 'wb') as f:
                        pickle.dump(r_mat_freeacc, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                r_mat = r_mat_freeacc[imu_name]

            # 90deg rotation matrix around x axis
            # r_mat = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0]])
            # multiply rotation matrix with opensim freeAcc global to get simulated freeAcc
            freeacc_g_opensim_simulated = self.simulate_freeacc(freeacc_g_opensim_point, r_mat)
            #  simulate Gyro
            R_euler_Mocap = np.flip(imus_coordinate[euler_trans_num]['r_eulor'])
            r_mat = euler2mat(R_euler_Mocap[0], R_euler_Mocap[1], R_euler_Mocap[2], axes='szyx')
            gyr_l_opensim_simulated = self.simulate_gyro(gyr_l_opensim_body, r_mat)
            # plot_imu_osimimu(imu_name, gyr_s_imu, gyr_l_opensim_simulated, freeacc_s_imu, freeacc_g_opensim_simulated)
        pass

    def simulate_freeacc(self, freeacc_g_opensim_point, r_mat):
        freeacc_g_opensim_simulated = np.dot(r_mat, freeacc_g_opensim_point.transpose()).transpose()
        return freeacc_g_opensim_simulated

    def simulate_gyro(self, gyr_l_opensim_body, r_mat):
        gyr_l_opensim_simulated = np.dot(gyr_l_opensim_body, r_mat) # % = (r_mat'*gyr_l_opensim')'
        return gyr_l_opensim_simulated

    def export_simulated_imu(self, osim_imu, osim_imu_result_folder):
        if not os.path.exists(osim_imu_result_folder):
            os.makedirs(osim_imu_result_folder)
        for key in osim_imu:
            osim_imu_file = osim_imu_result_folder + '/' + key + '.txt'
            osim_imu[key].to_csv(osim_imu_file, index=False)

    def get_imu_coordinates(self, model):
        state = model.initSystem()
        m1 = 'PelvisIMU'
        m2 = 'RPSI'
        m3 = 'LPSI'
        p1 = model.getMarkerSet().get(m1).get_location()
        p1 = np.array([p1[i] for i in range(len(p1))])
        p2 = model.getMarkerSet().get(m2).get_location()
        p2 = np.array([p2[i] for i in range(len(p2))])
        p3 = model.getMarkerSet().get(m3).get_location()
        p3 = np.array([p3[i] for i in range(len(p3))])
        p4 = (p2 + p3) / 2
        p4[2] = p1[2]
        x = p4 - p1
        y_temp = p2 - p1
        z = np.cross(y_temp, x)
        y = np.cross(z, x)
        x_norm = []
        y_norm = []
        z_norm = []
        for i in range(len(x)):
            x_norm.append(x[i] / np.linalg.norm(x))
            y_norm.append(y[i] / np.linalg.norm(y))
            z_norm.append(z[i] / np.linalg.norm(z))
        x_norm = np.array(x_norm)
        y_norm = np.array(y_norm)
        z_norm = np.array(z_norm)
        q_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        q_imu = np.array([x_norm, y_norm, z_norm]).round(4)
        r_mat = np.linalg.inv(q_imu).round(4)
        r_euler = np.flip(mat2euler(r_mat, axes='szyx'))
        # r_mat = Rotation.from_matrix(r_mat)
        # r_euler = np.flip(r_mat.as_euler('zyx'))
        Pelvis_imu_coord = {'r_eulor': r_euler.round(4), 'translation': p1.round(4)}
        # RIGHT UP LEG
        m1 = 'RUpLegIMU'
        m2 = 'RTS3'
        m3 = 'RTS2'
        m4 = 'RTS4'
        p1 = model.getMarkerSet().get(m1).get_location()
        p1 = np.array([p1[i] for i in range(len(p1))])
        p2 = model.getMarkerSet().get(m2).get_location()
        p2 = np.array([p2[i] for i in range(len(p2))])
        p3 = model.getMarkerSet().get(m3).get_location()
        p3 = np.array([p3[i] for i in range(len(p3))])
        p4 = model.getMarkerSet().get(m4).get_location()
        p4 = np.array([p4[i] for i in range(len(p4))])
        x = p3 - p2
        y = p4 - p2
        z = np.cross(x, y)
        x_norm = []
        y_norm = []
        z_norm = []
        for i in range(len(x)):
            x_norm.append(x[i] / np.linalg.norm(x))
            y_norm.append(y[i] / np.linalg.norm(y))
            z_norm.append(z[i] / np.linalg.norm(z))
        x_norm = np.array(x_norm)
        y_norm = np.array(y_norm)
        z_norm = np.array(z_norm)
        q_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        q_imu = np.array([x_norm, y_norm, z_norm])
        r_mat = np.linalg.inv(q_imu).round(4)
        r_euler = np.flip(mat2euler(r_mat, axes='szyx'))
        # r_mat = Rotation.from_matrix(r_mat)
        # r_euler = np.flip(r_mat.as_euler('zyx'))
        RUpLeg_imu_coord = {'r_eulor': r_euler.round(4), 'translation': p1.round(4)}
        ## RIGHT LO LEG
        m1 = 'RLoLegIMU'
        m2 = 'RSS3'
        m3 = 'RSS2'
        m4 = 'RSS4'
        p1 = model.getMarkerSet().get(m1).get_location()
        p1 = np.array([p1[i] for i in range(len(p1))])
        p2 = model.getMarkerSet().get(m2).get_location()
        p2 = np.array([p2[i] for i in range(len(p2))])
        p3 = model.getMarkerSet().get(m3).get_location()
        p3 = np.array([p3[i] for i in range(len(p3))])
        p4 = model.getMarkerSet().get(m4).get_location()
        p4 = np.array([p4[i] for i in range(len(p4))])

        x = p3 - p2
        y = p4 - p2
        z = np.cross(x, y)
        x_norm = []
        y_norm = []
        z_norm = []
        for i in range(len(x)):
            x_norm.append(x[i] / np.linalg.norm(x))
            y_norm.append(y[i] / np.linalg.norm(y))
            z_norm.append(z[i] / np.linalg.norm(z))
        x_norm = np.array(x_norm)
        y_norm = np.array(y_norm)
        z_norm = np.array(z_norm)
        q_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        q_imu = np.array([x_norm, y_norm, z_norm])
        r_mat = np.linalg.inv(q_imu).round(4)
        r_euler = np.flip(mat2euler(r_mat, axes='szyx'))
        # r_mat = Rotation.from_matrix(r_mat)
        # r_euler = np.flip(r_mat.as_euler('zyx'))
        RLoLeg_imu_coord = {'r_eulor': r_euler.round(4), 'translation': p1.round(4)}

        # Right Foot
        m1 = 'RFootIMU'
        m2 = 'RAnkleMed'
        m3 = 'RAnkleLat'

        p1 = model.getMarkerSet().get(m1).getLocationInGround(state)
        p1 = np.array([p1[i] for i in range(len(p1))])
        p2 = model.getMarkerSet().get(m2).getLocationInGround(state)
        p2 = np.array([p2[i] for i in range(len(p2))])
        p3 = model.getMarkerSet().get(m3).getLocationInGround(state)
        p3 = np.array([p3[i] for i in range(len(p3))])
        p4 = (p2 + p3) / 2

        p1_imu = model.getMarkerSet().get(m1).get_location()
        p1_imu = np.array([p1_imu[i] for i in range(len(p1_imu))])

        x = p4 - p1
        y_temp = p3 - p1
        z = np.cross(x, y_temp)
        y = np.cross(z, x)
        x_norm = []
        y_norm = []
        z_norm = []
        for i in range(len(x)):
            x_norm.append(x[i] / np.linalg.norm(x))
            y_norm.append(y[i] / np.linalg.norm(y))
            z_norm.append(z[i] / np.linalg.norm(z))
        x_norm = np.array(x_norm)
        y_norm = np.array(y_norm)
        z_norm = np.array(z_norm)
        q_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        q_imu = np.array([x_norm, y_norm, z_norm])
        r_mat = np.linalg.inv(q_imu).round(4)
        r_euler = np.flip(mat2euler(r_mat, axes='szyx'))
        # r_mat = Rotation.from_matrix(r_mat)
        # r_euler = np.flip(r_mat.as_euler('zyx'))
        RFoot_imu_coord = {'r_eulor': r_euler.round(4), 'translation': p1_imu.round(4)}

        # LEFT UP LEG
        m1 = 'LUpLegIMU'
        m2 = 'LTS3'
        m3 = 'LTS2'
        m4 = 'LTS4'
        p1 = model.getMarkerSet().get(m1).get_location()
        p1 = np.array([p1[i] for i in range(len(p1))])
        p2 = model.getMarkerSet().get(m2).get_location()
        p2 = np.array([p2[i] for i in range(len(p2))])
        p3 = model.getMarkerSet().get(m3).get_location()
        p3 = np.array([p3[i] for i in range(len(p3))])
        p4 = model.getMarkerSet().get(m4).get_location()
        p4 = np.array([p4[i] for i in range(len(p4))])
        x = p3 - p2
        y = -(p4 - p2)
        z = np.cross(x, y)
        x_norm = []
        y_norm = []
        z_norm = []
        for i in range(len(x)):
            x_norm.append(x[i] / np.linalg.norm(x))
            y_norm.append(y[i] / np.linalg.norm(y))
            z_norm.append(z[i] / np.linalg.norm(z))
        x_norm = np.array(x_norm)
        y_norm = np.array(y_norm)
        z_norm = np.array(z_norm)
        q_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        q_imu = np.array([x_norm, y_norm, z_norm])
        r_mat = q_g / q_imu
        r_mat = np.linalg.inv(q_imu).round(4)
        r_euler = np.flip(mat2euler(r_mat, axes='szyx'))
        # r_mat = Rotation.from_matrix(r_mat)
        # r_euler = np.flip(r_mat.as_euler('zyx'))
        LUpLeg_imu_coord = {'r_eulor': r_euler.round(4), 'translation': p1.round(4)}
        ## LEFT LO LEG
        m1 = 'LLoLegIMU'
        m2 = 'LSS3'
        m3 = 'LSS2'
        m4 = 'LSS4'
        p1 = model.getMarkerSet().get(m1).get_location()
        p1 = np.array([p1[i] for i in range(len(p1))])
        p2 = model.getMarkerSet().get(m2).get_location()
        p2 = np.array([p2[i] for i in range(len(p2))])
        p3 = model.getMarkerSet().get(m3).get_location()
        p3 = np.array([p3[i] for i in range(len(p3))])
        p4 = model.getMarkerSet().get(m4).get_location()
        p4 = np.array([p4[i] for i in range(len(p4))])

        x = p3 - p2
        y = -(p4 - p2)
        z = np.cross(x, y)
        x_norm = []
        y_norm = []
        z_norm = []
        for i in range(len(x)):
            x_norm.append(x[i] / np.linalg.norm(x))
            y_norm.append(y[i] / np.linalg.norm(y))
            z_norm.append(z[i] / np.linalg.norm(z))
        x_norm = np.array(x_norm)
        y_norm = np.array(y_norm)
        z_norm = np.array(z_norm)
        q_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        q_imu = np.array([x_norm, y_norm, z_norm])
        r_mat = np.linalg.inv(q_imu).round(4)
        r_euler = np.flip(mat2euler(r_mat, axes='szyx'))
        # r_mat = Rotation.from_matrix(r_mat)
        # r_euler = np.flip(r_mat.as_euler('zyx'))
        LLoLeg_imu_coord = {'r_eulor': r_euler.round(4), 'translation': p1.round(4)}

        # LEFT Foot
        m1 = 'LFootIMU'
        m2 = 'LAnkleMed'
        m3 = 'LAnkleLat'

        p1 = model.getMarkerSet().get(m1).getLocationInGround(state)
        p1 = np.array([p1[i] for i in range(len(p1))])
        p2 = model.getMarkerSet().get(m2).getLocationInGround(state)
        p2 = np.array([p2[i] for i in range(len(p2))])
        p3 = model.getMarkerSet().get(m3).getLocationInGround(state)
        p3 = np.array([p3[i] for i in range(len(p3))])
        p4 = (p2 + p3) / 2

        p1_imu = model.getMarkerSet().get(m1).get_location()
        p1_imu = np.array([p1_imu[i] for i in range(len(p1_imu))])

        x = p4 - p1
        y_temp = p3 - p1
        z = np.cross(y_temp, x)
        y = np.cross(z, x)
        x_norm = []
        y_norm = []
        z_norm = []
        for i in range(len(x)):
            x_norm.append(x[i] / np.linalg.norm(x))
            y_norm.append(y[i] / np.linalg.norm(y))
            z_norm.append(z[i] / np.linalg.norm(z))
        x_norm = np.array(x_norm)
        y_norm = np.array(y_norm)
        z_norm = np.array(z_norm)
        q_g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        q_imu = np.array([x_norm, y_norm, z_norm])
        r_mat = np.linalg.inv(q_imu).round(4)
        r_euler = np.flip(mat2euler(r_mat, axes='szyx'))
        # r_mat = Rotation.from_matrix(r_mat)
        # r_euler = np.flip(r_mat.as_euler('zyx'))
        LFoot_imu_coord = {'r_eulor': r_euler.round(4), 'translation': p1_imu.round(4)}
        imus_coordinate = [Pelvis_imu_coord, RUpLeg_imu_coord, RLoLeg_imu_coord, RFoot_imu_coord,
                           LUpLeg_imu_coord, LLoLeg_imu_coord, LFoot_imu_coord]
        return imus_coordinate