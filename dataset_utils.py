import sys
sys.path.insert(0, '../collab_radar_eval')
# from config import *
import matplotlib.pyplot as plt
import matplotlib
import json
import numpy as np
import os
from PIL import Image
from shutil import copyfile
import pdb

def generate_extrinsic_from_rot_tran(angle, translation):

    extrinsic_matrix = np.eye(4)
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle), 0],
                                [-np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    extrinsic_matrix[:3,:3] = rotation_matrix
    extrinsic_matrix[:3,3] = np.array(translation)

    return extrinsic_matrix

class collab_dataset():

    def __init__(self, base_path, gt_folder_name, bev_folder_name):

        self.base_path = base_path
        self.gt_folder_name = gt_folder_name
        self.bev_folder_name = bev_folder_name


    def get_image(self, timestamp, veh_id):

        veh_id = int(veh_id)
        img = plt.imread(f"{self.base_path}/radar_bev_images/{self.bev_folder_name}/plot_data_veh{str(veh_id)}_{str(timestamp)}.jpg")
        return np.array(img)


    def get_extrinsic(self, timestamp, veh_id, error=False):
        
        veh_id = int(veh_id)

        with open(f'{self.base_path}/ground_truth/{self.gt_folder_name}/labels_{timestamp}.json', 'r') as file:
            label_dict = json.load(file)

        angle = np.deg2rad(label_dict[str(veh_id)]["yaw"])
        translation = label_dict[str(veh_id)]["center"] 
        if error:
            translation[:2] = translation[:2] + np.random.rand(2,)*0
        # translation[:2] = translation[:2] + np.array([1,1]) 
        # angle = -angle
        rot = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
        translation[:2] = translation[:2] + np.array([0.0,4.0])@rot ## radar translation

        extrinsic = generate_extrinsic_from_rot_tran(angle, translation)

        return extrinsic

    def get_extrinsic_car1_to_car2(self, timestamp, car_1_id, car_2_id):

        car1_to_world = self.get_extrinsic(timestamp, car_1_id, True)
        car2_to_world = self.get_extrinsic(timestamp, car_2_id, True)

        car1_to_car2 = np.linalg.inv(car2_to_world)@car1_to_world
        return car1_to_car2

    def get_nearby_vehicles_at_time(self, veh_id, timestamp, v2v_max_distance):

        veh_id = int(veh_id)

        labels = self.get_gt_wrt_local_for_time(timestamp, veh_id)
    
        labels_for_v2v = labels[np.linalg.norm(labels[:,1:3],axis=1)<v2v_max_distance]

        extrinsics = []
        for collab_veh_id in list(labels_for_v2v[:,0]):
            extrinsics.append(self.get_extrinsic(timestamp,int(collab_veh_id)))

        return labels_for_v2v, np.array(extrinsics)   

    def get_label_dict_world_coord_for_timestamp(self,timestamp):

        # veh_id = int(veh_id)
        
        with open(f'{self.base_path}/ground_truth/{self.gt_folder_name}/labels_{timestamp}.json', 'r') as file:
            label_dict = json.load(file)

        return label_dict

    def get_gt_wrt_local_for_time(self, timestamp, veh_id):

        veh_id = int(veh_id)

        with open(f'{self.base_path}/ground_truth/{self.gt_folder_name}/labels_{timestamp}.json', 'r') as file:
            label_dict = json.load(file)

        angle_ego = label_dict[str(veh_id)]["yaw"]
        # translation = label_dict[str(veh_id)]["center"]

        extrinsic_ego2world = self.get_extrinsic(timestamp, veh_id, False)
        # print(extrinsic)

        label_boxes = []

        for key, val in label_dict.items():

            if key!=str(veh_id):

                label = np.zeros(8,) #[id,x,y,z,l,w,h,theta] #z is height
                label[0] = int(key)
                label[1:4] = np.matmul(np.linalg.inv(extrinsic_ego2world),np.hstack((val["center"],1)))[:3]
                label[4:7] = val["dimensions"]
                label[7] = val["yaw"] - angle_ego
                label_boxes.append(label)

        if not label_boxes:
            return np.zeros((0,8))
        return np.array(label_boxes)

    def plot_labels(self, labels):
        fig2, ax = plt.subplots(1,num=2)
                # fig2.clf()
        plt.xlim([-1000, 1000])
        plt.ylim([-1000, 1000])
        plt.title(' Ground Truth ')

        for n_lbl in range(labels.shape[0]):
            ts = ax.transData
            center_coords = labels[n_lbl,:3]
            dimensions = labels[n_lbl,3:6]
            dimensions[1] = 10
            angle = labels[n_lbl,6]

            # tr = matplotlib.transforms.Affine2D().rotate_deg_around(center_coords[0], center_coords[1], -angle)
            # t = tr + ts

            box = patches.Rectangle((center_coords[0]-(dimensions[0]*0.5), center_coords[1]-(dimensions[1]*0.5)), dimensions[0], dimensions[1], angle = -angle,
             linewidth=2, edgecolor='r', facecolor="r")
            ax.add_patch(box)
            ax.quiver(center_coords[0], center_coords[1],np.sin(np.deg2rad(angle))\
                       ,np.cos(np.deg2rad(angle)), scale=40, headwidth=2, width=0.003, headlength=3)#, color=colors)

        # plt.savefig(f'{out_paths["path_gt"]}/sample_plot.jpg')

    def resize(self, rad_img, corners, img_size = 1152, first = False):
        
        scale_rat = img_size / rad_img.shape[0]
        rad_img_Im = Image.fromarray(rad_img)
        rad_img_Im = rad_img_Im.resize((img_size, img_size))
        rad_img = np.array(rad_img_Im)

        if corners is not None:
            corners = corners * scale_rat
        
        if first:
            self.res /= scale_rat
        
        return rad_img, corners

    def plot_radar_scene(self, rad_img, annotations = None, ax = None, fig = None, save = False, id_ = None, scale = 1152):
        '''
        annotations = [[x,z,y,l,w,h,a]]
        #[id,x,y,z,w,l,h,theta]
        '''
        # print(annotations)
        bev_size = 128
        limit = 80
        res = 80/128
        rad_img_down, _ = self.resize(rad_img, None, bev_size)
        ax.imshow(rad_img_down)

        #Filter annotations
        annotations = annotations[(annotations[:,1]<limit/2)*(annotations[:,1]>-limit/2)]
        annotations = annotations[(annotations[:,2]<limit)*(annotations[:,2]>0)]

        # annotations = annotations[4,:]
        # print(annotations)


        if annotations is not None and annotations.shape[0] != 0:
            if len(annotations.shape) == 1:
                annotations = annotations.reshape((1,-1))
            l = annotations[:,4] / res
            w = annotations[:,5] / res
            a = np.deg2rad(annotations[:,7])

            # corners = np.array([
            #     [-l/2,-w/2],[-l/2,w/2],[l/2,-w/2],[l/2,w/2]
            # ]).transpose((2,0,1))

            corners = np.array([
                [-w/2,-l/2],[-w/2,l/2],[w/2,-l/2],[w/2,l/2]
            ]).transpose((2,0,1))


            rots = np.array([
                [np.cos(a), -np.sin(a)],
                [np.sin(a), np.cos(a)]
            ]).transpose((2,0,1))
            
            rot_corners = corners @ rots 
            
            offset = annotations[:,1:3]
            # offset[:,1] = limit - offset[:,1]
            # offset[:,0] = offset[:,0] + limit/2
            # offset = offset / res
            rot_corners = rot_corners + offset.reshape((-1,1,2))/res

            rot_corners[:,:,1] = limit/res - rot_corners[:,:,1]
            rot_corners[:,:,0] = limit/2/res + rot_corners[:,:,0]

            # _, rot_corners = self.resize(rad_img, rot_corners, scale)
            
            for box in rot_corners:
                ax.plot([box[1,0], box[0,0]], [box[1,1], box[0,1]],color='red')
                ax.plot([box[2,0], box[0,0]], [box[2,1], box[0,1]],  color='red')
                ax.plot([box[3,0], box[1,0]], [box[3,1], box[1,1]], color='red')
                ax.plot([box[3,0], box[2,0]], [box[3,1], box[2,1]], color='red')

if __name__ == '__main__':
    
    base_path = f"/media/ehdd_8t1/RadarImaging/collab_radar_data/"
    gt_folder_name = "downtown_SD_10thru_50count_labels"
    bev_folder_name = "downtown_SD_10thru_50count_80m_doppler_tuned"
    
    collab_dataset = collab_dataset(base_path, gt_folder_name, bev_folder_name)

    timestamp = 70
    veh_id = 15
    rad_img = collab_dataset.get_image(timestamp = timestamp, veh_id = veh_id)
    labels = collab_dataset.get_gt_wrt_local_for_time(timestamp = timestamp, veh_id = veh_id)

    fig2, ax = plt.subplots(1)
    collab_dataset.plot_radar_scene(rad_img, labels, ax, fig = None, save = False, id_ = None, scale = 1152)
    plt.savefig(f'sample_plot.jpg')


    # print("e")
