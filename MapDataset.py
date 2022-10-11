import torch.utils.data as torchdata
import pandas as pd
from shapely.geometry import Polygon, Point
import os
import json
import numpy as np

from util import *

gt_folder_name = "downtown_SD_10thru_50count_labels"
bev_folder_name = "downtown_SD_10thru_50count_80m_doppler_tuned"


class MapDataset(torchdata.Dataset):
    def __init__(self, car_file, polygon_file, radar_dataset="", *args, **kwargs):
        df = pd.read_csv(car_file)
        if df["vehicle_angle"].max() > 2 * np.pi or df["vehicle_angle"].min() < 0:
            df["vehicle_angle"] = df["vehicle_angle"] * np.pi / 180

        self.df = df
        self.vehicle_names = list(df["vehicle_id"].unique())[:40]

        self.dx, self.bx, (self.nx, self.ny) = get_grid([4, -40,
                                                         84, 40],
                                                        [80 / 128, 80 / 128])

        with open(polygon_file, "r") as file:
            self.polygon = json.load(file)

        # with open(os.path.join(radar_dataset, "training_txt_files", bev_folder_name, "valid_files_train.txt"),
        #           'r') as file:
        #     training = file.readlines()
        #
        # with open(os.path.join(radar_dataset, "training_txt_files", bev_folder_name, "valid_files_test.txt"),
        #           'r') as file:
        #     testing = file.readlines()
        training = ["plot_data_veh129_382", "plot_data_veh115_395", "plot_data_veh122_396"]
        testing = ["plot_data_veh129_382", "plot_data_veh115_395", "plot_data_veh122_396"]
        self.radar_data = training + testing
        self.ixes = self.get_ixes()

    def get_ixes(self):
        ixes = []
        for data in self.radar_data:
            _, _, name, time = data.split("_")
            ixes.append((name, int(time)))
        return ixes

    def render(self, name, time):
        data = self.df[(self.df['vehicle_id'] == name) & (self.df['timestep_time'] == time)]
        x = data['vehicle_x']
        y = data['vehicle_y']
        angle = data['vehicle_angle']

        others = self.df[self.df['timestep_time'] == time]
        pos = others[["vehicle_x", "vehicle_y"]].values - np.array(x, y)
        mask = np.linalg.norm(pos, axis=1) < 100

        others = others[mask]

        center = np.array([x, y, np.cos(angle), np.sin(angle)]).flatten()
        # other_x = others['vehicle_x'].values.reshape(-1, 1)
        # other_y = others['vehicle_y'].values.reshape(-1, 1)
        # other_angle = others['vehicle_angle'].values.reshape(-1, 1)
        #
        # objs = np.hstack([other_x, other_y, np.cos(other_angle), np.sin(other_angle)])
        # if len(objs) == 0:
        #     lobjs = np.zeros((0, 4))
        # else:
        #     lobjs = objects2frame(objs[np.newaxis, :, :], center)[0]
        #
        # # create image of other objects
        # obj_img = np.zeros((self.nx, self.ny))
        # for box in lobjs:
        #     pts = get_corners(box, [1.73, 4.084][::-1])
        #     pts = np.round(
        #         (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
        #     ).astype(np.int32)
        #     pts[:, [1, 0]] = pts[:, [0, 1]]
        #     cv2.fillPoly(obj_img, [pts], 1.0)

        # create image of ego
        center_img = np.zeros((self.nx, self.ny))
        pts = get_corners([0.0, 0.0, 1.0, 0.0], [1.73, 4.084][::-1])
        pts = np.round(
            (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
        ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(center_img, [pts], 1.0)

        polygon_list = {}
        for poly_type, value in self.polygon.items():
            if isinstance(value, list): continue
            polygon_list[poly_type] = {}
            for key, shape in value.items():
                if poly_type == 'lane_markings' and key in polygon_list['lane']:
                    for i in range(len(shape)):
                        polygon_list[poly_type][key + str(i)] = np.asarray(shape[i])
                elif poly_type == 'lane_markings':
                    continue
                else:
                    shape = np.asarray(shape)

                    if any(np.linalg.norm(shape - center[:2], axis=1) <= 100):
                        polygon_list[poly_type][key] = np.asarray(shape)

        map_img = np.zeros((self.nx, self.ny))

        def draw_polygon(shapes, image, fill=False):
            angle = -np.arctan2(center[3], center[2]) + np.pi / 2
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
            for key, poly in shapes.items():
                poly = np.dot(poly - np.array([center[0], center[1]]), rot)
                pts = np.round(
                    (poly - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                ).astype(np.int32)
                pts = np.clip(pts, 0, self.nx)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                if fill:
                    cv2.fillPoly(image, [pts], color=1.0)
                else:
                    cv2.polylines(image, [pts], isClosed=False, color=1.0)

        draw_polygon(polygon_list['junction'], map_img, fill=True)
        draw_polygon(polygon_list['lane'], map_img, fill=True)

        # road_div = np.zeros((self.nx, self.ny))
        # draw_polygon(polygon_list['junction'], road_div)
        # draw_polygon(polygon_list['lane'], road_div)

        # lane_div = np.zeros((self.nx, self.ny))
        # x = np.stack([map_img, lane_div, road_div, obj_img, center_img])
        return map_img[np.newaxis, :, :]

    def __getitem__(self, index):
        name, time = self.ixes[index]

        return self.render(name, time)


if __name__ == "__main__":
    dataset = MapDataset("Data/downtown_SD_10thru_50count_with_cad_id.csv", "Data/polygons.json")
    for x in iter(dataset):
        print(x.shape)
        render_observation(x)
        plt.show()

