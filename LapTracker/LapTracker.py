# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:15:09 2020

@author: Adrian
"""
import numpy as np
import pandas as pd
import numpy.matlib
import networkx as nx
import warnings
from progress.bar import Bar
from skimage.measure import regionprops_table
from scipy.spatial import distance
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


class LapTracker():
    """
    A class used to represent an a linear assignment problem tracker

    The tracking algorithm used is based on

    ...

    Attributes
    ----------
    df : pd.DataFrame
        a pandas dataframe containing the x, y, and t coordinates of objects
    max_distance : int
        maximal distance to allow particle linking between frames
    time_window : int
        maximal memory of the tracker to link track segments
    identifiers : list
        list of column names for coordinates in df and label column
        (e.g. ['x_coord', 'y_coord', 'timepoint', 'labels'])
    max_split_distance : int
        maximal distance to link segment starts to segment middlepoints
    allow_merging : bool
        indicates whether object merging is allowed or not (default: False)
    allow_splitting: bool
        indicates whether object splitting is allowed or not (default: True)

    Methods
    -------
    track()
        tracks the objects in df and assigns 3 additional collumns
        (unique_id, segment_id and track_id).
    """

    def __init__(self,
                 max_distance,
                 time_window,
                 max_split_distance,
                 allow_merging=False,
                 allow_splitting=True):
        """
        Parameters
        ----------
        max_distance : int
            maximal distance to allow particle linking between frames
        time_window : int
            maximal memory of the tracker to link track segments
        max_split_distance : int
            maximal distance to link segment starts to segment middlepoints
        allow_merging : bool
            indicates whether object merging is allowed or not
            (default: False)
        allow_splitting: bool
            indicates whether object splitting is allowed or not
            (default: True)
        """

        self.max_distance = max_distance
        self.max_split_distance = max_split_distance
        self.time_window = time_window
        self.allow_merging = allow_merging
        self.allow_splitting = allow_splitting
        self.link_costs = []
        self.global_costs = []

    def __get_frame_linking_matrix(self,
                                   t_coords,
                                   t1_coords):
        """Generates cost matrix for object linking"""

        number_of_objects_t_0 = len(t_coords)
        number_of_objects_t_1 = len(t1_coords)
        n_objects = number_of_objects_t_0 + number_of_objects_t_1
        frame_linking_matrix = np.zeros([n_objects, n_objects])
        # calculate distance matrix/linking matrix
        dist = distance_matrix(t_coords, t1_coords)
        # replace entries that are above threshold
        dist[dist > self.max_distance] = np.inf
        frame_linking_matrix[0:number_of_objects_t_0,
                             0:number_of_objects_t_1] = dist
        # calculate the lower right block
        lower_right_block = np.transpose(dist)*0.0001
        frame_linking_matrix[
            number_of_objects_t_0:n_objects,
            number_of_objects_t_1:n_objects] = lower_right_block
        # calculate non-link matrix
        if self.link_costs == []:
            non_link_cost = self.max_distance
        else:
            non_link_cost = 1.05*np.max(self.link_costs)

        non_link = np.ones([number_of_objects_t_0,
                            number_of_objects_t_0], int)*np.inf
        np.fill_diagonal(non_link, non_link_cost)
        frame_linking_matrix[0:number_of_objects_t_0,
                             number_of_objects_t_1:n_objects] = non_link

        non_link_1 = np.ones([number_of_objects_t_1,
                              number_of_objects_t_1], int)*np.inf
        np.fill_diagonal(non_link_1, non_link_cost)
        frame_linking_matrix[number_of_objects_t_0:n_objects,
                             0:number_of_objects_t_1] = non_link_1

        return frame_linking_matrix

    def __get_segment_linking_matrix(self):
        """Generates cost matrix for segment linking"""

        segment_linking_matrix = np.ones([
            (self.number_of_segments +
             self.number_of_segment_middlepoints)*2,
            (self.number_of_segments +
             self.number_of_segment_middlepoints)*2]) * np.inf

        # gap_closing_matrix

        dist = distance_matrix(self.end_coordinates[:, 0:2],
                               self.start_coordinates[:, 0:2])
        temp_dist = distance_matrix(self.end_coordinates[:, 2:4],
                                    self.start_coordinates[:, 2:4])
        disallowed_dist = temp_dist > self.time_window
        dist[disallowed_dist] = np.inf

        # replace entries that are above threshold

        dist[dist > self.max_distance] = np.inf

        gap_closing_matrix = dist

        # replace diagonal with inf

        np.fill_diagonal(gap_closing_matrix, np.inf)

        segment_linking_matrix[0:self.number_of_segments,
                               0:self.number_of_segments] = gap_closing_matrix

        self.global_costs.extend(list(
            segment_linking_matrix[segment_linking_matrix < np.inf]))

        # merge matrix

        merge_matrix = np.ones([self.number_of_segments,
                                self.number_of_segment_middlepoints])*np.inf

        segment_linking_matrix[0:self.number_of_segments,
                               self.number_of_segments:
                                   (self.number_of_segment_middlepoints +
                                    self.number_of_segments)] = merge_matrix

        self.global_costs.extend(list(merge_matrix[merge_matrix < np.inf]))

        # center_matrix

        center_matrix = np.ones([self.number_of_segment_middlepoints,
                                 self.number_of_segment_middlepoints])*np.inf

        segment_linking_matrix[self.number_of_segments:
                               (self.number_of_segment_middlepoints +
                                self.number_of_segments),
                               self.number_of_segments:
                                   (self.number_of_segment_middlepoints +
                                    self.number_of_segments)] = center_matrix

        # split_matrix

        split_dist = distance_matrix(self.segment_middlepoints.iloc[:, 0:2],
                                     self.start_coordinates[:, 0:2])

        split_dist[split_dist == 0] = np.inf

        # find splits that are close enough at t-1

        matrix_a = np.transpose(
            numpy.matlib.repmat(self.segment_middlepoints.iloc[:, 2],
                                self.number_of_segments, 1))
        matrix_b = numpy.matlib.repmat(self.start_coordinates[:, 2],
                                       self.number_of_segment_middlepoints, 1)

        matrix_c = matrix_b - matrix_a

        split_dist[(matrix_c > 1) | (matrix_c <= 0)] = np.inf
        split_dist[split_dist > self.max_split_distance] = np.inf

        segment_linking_matrix[
            self.number_of_segments:(self.number_of_segments +
                                     self.number_of_segment_middlepoints),
            0:self.number_of_segments] = split_dist

        self.global_costs.extend(list(split_dist[split_dist < np.inf]))

        # termination matrix

        termination_matrix = np.ones(
            [self.number_of_segments,
             (self.number_of_segments +
              self.number_of_segment_middlepoints)])*np.inf

        self.global_costs = np.array(self.global_costs)
        termination_cost = np.quantile(self.global_costs, 0.9)

        np.fill_diagonal(termination_matrix, termination_cost)

        segment_linking_matrix[
            0:self.number_of_segments,
            (self.number_of_segments +
             self.number_of_segment_middlepoints):] = termination_matrix

        # initiation matrix

        initiation_matrix = np.ones([(self.number_of_segments +
                                      self.number_of_segment_middlepoints),
                                     self.number_of_segments])*np.inf

        np.fill_diagonal(initiation_matrix, termination_cost)

        segment_linking_matrix[(self.number_of_segments +
                                self.number_of_segment_middlepoints):,
                               0:self.number_of_segments] = initiation_matrix

        # merge_refusal_matrix

        merge_refusal_matrix = np.ones(
            [(self.number_of_segments +
              self.number_of_segment_middlepoints),
             self.number_of_segment_middlepoints])*np.inf

        allowed = np.ones([self.number_of_segment_middlepoints,
                           self.number_of_segment_middlepoints])*np.inf

        diagonal = self.average_displacement[np.array(
            [self.segment_middlepoints.iloc[:, 4]])]**2

        np.fill_diagonal(allowed, diagonal)

        merge_refusal_matrix[self.number_of_segments:,
                             0:] = allowed

        segment_linking_matrix[(
            self.number_of_segments + self.number_of_segment_middlepoints):,
            self.number_of_segments:(
                self.number_of_segments +
                self.number_of_segment_middlepoints)] = merge_refusal_matrix

        split_refusal_matrix = np.ones(
            [self.number_of_segment_middlepoints,
             (self.number_of_segments +
              self.number_of_segment_middlepoints)])*np.inf

        split_refusal_matrix[0:,
                             self.number_of_segments:] = allowed

        segment_linking_matrix[
            self.number_of_segments:(self.number_of_segment_middlepoints +
                                     self.number_of_segments),
            (self.number_of_segments +
             self.number_of_segment_middlepoints):] = split_refusal_matrix

        # get lower right block

        lower_right_block = segment_linking_matrix[
            0:(self.number_of_segments +
               self.number_of_segment_middlepoints),
            0:(self.number_of_segments +
               self.number_of_segment_middlepoints)]

        lower_right_block = np.transpose(lower_right_block)*0.0001

        segment_linking_matrix[
            (self.number_of_segments +
             self.number_of_segment_middlepoints):,
            (self.number_of_segments +
             self.number_of_segment_middlepoints):] = lower_right_block

        return segment_linking_matrix

    def __get_track_segments(self):
        """Computes track segments from segment linking matrix"""
        bar = Bar('linking objects across time', max=self.number_of_timepoints,
                  check_tty=False, hide_cursor=False)
        for timepoint in range(0, self.number_of_timepoints):

            features_t0 = self.df.loc[
                self.df[self.identifiers[2]] == timepoint].sort_values(
                    'unique_id')
            features_t1 = self.df.loc[
                self.df[self.identifiers[2]] == timepoint + 1].sort_values(
                    'unique_id')

            number_of_objects_t_0 = len(features_t0)
            number_of_objects_t_1 = len(features_t1)

            if (number_of_objects_t_0 != 0) & (number_of_objects_t_1 != 0):

                t_coords = features_t0[[self.identifiers[0],
                                        self.identifiers[1]]]

                t1_coords = features_t1[[self.identifiers[0],
                                         self.identifiers[1]]]
                # get cost matrix

                self.cost_matrix_linking = self.__get_frame_linking_matrix(
                    t_coords,
                    t1_coords)

                # get optimal linking from cost matrix
                row_ind, col_ind = linear_sum_assignment(
                    self.cost_matrix_linking)

                matches = np.where(col_ind < number_of_objects_t_1)[0]

                for match in matches:
                    if ((col_ind[match] < number_of_objects_t_1) &
                        (row_ind[match] < number_of_objects_t_0)):
                        self.adjacency_matrix[
                            features_t0['unique_id'].iloc[row_ind[match]],
                            features_t1['unique_id'].iloc[col_ind[match]]] = 1

                link_matrix = self.cost_matrix_linking[
                    0: number_of_objects_t_0,
                    0: number_of_objects_t_1]

                self.object_row_index = row_ind
                self.object_col_index = col_ind

                # add costs of the links made at this timepoint
                self.link_costs = self.link_costs + list(
                    link_matrix[link_matrix < np.inf])
            bar.next()
        bar.finish()
        # compute a weakly connected directed graph from the
        # adjacency matrix. I used the graph approach because
        # it's relatively simple to get the single tracks out.

        self.G = nx.DiGraph(self.adjacency_matrix)
        self.number_of_segments = nx.number_weakly_connected_components(self.G)
        self.segments = [sorted(c) for c in sorted(
            nx.weakly_connected_components(self.G),
            key=len,
            reverse=True)]

        self.segments_by_label = [list(self.df[self.identifiers[3]].iloc[
            sorted(c)]) for c in sorted(nx.weakly_connected_components(self.G),
                                        key=len,
                                        reverse=True)]

        # add column for segment ids

        segment_ids = []
        for obj in range(0, self.number_of_objects):
            for segment_id, segment in enumerate(self.segments):
                if obj in segment:
                    segment_ids.append(segment_id)

        self.df['segment_id'] = segment_ids

    def __close_gaps(self):
        """Deals with gaps/merging/splitting based on to gap closing matrix"""

        # get start and end coordinates of the detected segments

        start_coordinates = [[self.df[self.identifiers[0]].iloc[sorted(c)[0]],
                              self.df[self.identifiers[1]].iloc[sorted(c)[0]],
                              self.df[self.identifiers[2]].iloc[sorted(c)[0]],
                              0]
                             for c in sorted(
                                     nx.weakly_connected_components(self.G),
                                     key=len,
                                     reverse=True)]

        end_coordinates = [[self.df[self.identifiers[0]].iloc[sorted(c)[-1]],
                            self.df[self.identifiers[1]].iloc[sorted(c)[-1]],
                            self.df[self.identifiers[2]].iloc[sorted(c)[-1]],
                            0]
                           for c in sorted(
                                   nx.weakly_connected_components(self.G),
                                   key=len,
                                   reverse=True)]

        self.start_coordinates = np.array(start_coordinates)
        self.end_coordinates = np.array(end_coordinates)

        # get the segment middle points

        segment_middlepoints = []
        for segment in self.segments:
            for segment_object in segment[1:-1]:
                segment_middlepoints.append([
                    self.df[self.identifiers[0]].iloc[segment_object],
                    self.df[self.identifiers[1]].iloc[segment_object],
                    self.df[self.identifiers[2]].iloc[segment_object],
                    self.df['unique_id'].iloc[segment_object],
                    self.df['segment_id'].iloc[segment_object]])

        segment_middlepoints = pd.DataFrame(np.array(segment_middlepoints),
                                            columns=[self.identifiers[0],
                                                     self.identifiers[1],
                                                     self.identifiers[2],
                                                     'unique_id',
                                                     'segment_id'],
                                            dtype='uint16')

        self.segment_middlepoints = segment_middlepoints.sort_values(
            'unique_id')
        self.number_of_segment_middlepoints = len(segment_middlepoints)

        # calculate average displacement for each segment
        # this is later used to compute the cost matrix

        self.average_displacement = []

        for segment_id in range(self.number_of_segments):
            current_segment_displacements = []
            current_segment = self.df.loc[self.df.segment_id == segment_id]
            for timepoint in current_segment[self.identifiers[2]]:
                if timepoint == np.max(current_segment[self.identifiers[2]]):
                    break
                else:
                    current_displacement = distance.euclidean(
                        current_segment[[self.identifiers[0],
                                         self.identifiers[1]]].
                        loc[current_segment[self.identifiers[2]] == timepoint],
                        current_segment[[self.identifiers[0],
                                         self.identifiers[1]]].
                        loc[current_segment[
                            self.identifiers[2]] == timepoint+1])
                    current_segment_displacements.append(current_displacement)
            self.average_displacement.append(
                np.mean(np.array(current_segment_displacements)))

        self.average_displacement = np.array(
            self.average_displacement)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            average_displacement_all_segments = np.nanmean(
                self.average_displacement)
            
        self.average_displacement[np.isnan(
            self.average_displacement)] = average_displacement_all_segments

        self.cost_matrix_gap_closing = self.__get_segment_linking_matrix()

        row_ind, col_ind = linear_sum_assignment(self.cost_matrix_gap_closing)

        # get unique ids of the middlepoints that undergo a split

        sources = np.where(col_ind < self.number_of_segments)[0]
        sources = sources[(sources > self.number_of_segments) &
                          (sources < (self.number_of_segments +
                                      self.number_of_segment_middlepoints))]
        target_segment_starts = col_ind[np.array(sources)]
        sources = sources - self.number_of_segments
        sources_unique_id = np.array(
            self.segment_middlepoints['unique_id'].iloc[sources], dtype='int')

        # get unique id of the segment starting points

        start_points = [[self.df[self.identifiers[0]].iloc[sorted(c)[0]],
                         self.df[self.identifiers[1]].iloc[sorted(c)[0]],
                         self.df['unique_id'].iloc[sorted(c)[0]],
                         self.df['segment_id'].iloc[sorted(c)[0]], 0]
                        for c in sorted(nx.weakly_connected_components(self.G),
                                        key=len,
                                        reverse=True)]

        end_points = [[self.df[self.identifiers[0]].iloc[sorted(c)[-1]],
                       self.df[self.identifiers[1]].iloc[sorted(c)[-1]],
                       self.df['unique_id'].iloc[sorted(c)[-1]],
                       self.df['segment_id'].iloc[sorted(c)[-1]], 0]
                      for c in sorted(nx.weakly_connected_components(self.G),
                                      key=len,
                                      reverse=True)]
        end_points = np.array(end_points)
        start_points = np.array(start_points)
        target_unique_id = np.array(
            start_points[np.array(target_segment_starts)][:, 2], dtype='int')

        self.adjacency_matrix[sources_unique_id, target_unique_id] = 1

        # gap closing

        sources = np.where(col_ind < self.number_of_segments)[0]
        sources = sources[(sources >= 0) &
                          (sources < self.number_of_segments)]
        sources_unique_id = np.array(
            end_points[np.array(sources)][:, 2], dtype='int')

        # get unique id of the segment starting points
        target_segment_starts = col_ind[np.array(sources)]
        target_unique_id = np.array(
            start_points[np.array(target_segment_starts)][:, 2], dtype='int')

        self.adjacency_matrix[sources_unique_id, target_unique_id] = 1
        self.segment_row_index = row_ind
        self.segment_col_index = col_ind

    def track_df(self, df, identifiers):
        """
        Tracks the objects in df

        Assigns 3 additional columns to df
        (unique_id, segment_id and track_id).

        Parameters
        ----------
        df : pd.DataFrame
            a pandas dataframe containing the x, y, and t coordinates
            of objects
        identifiers : list
            list of column names for coordinates in df and label column
            (e.g. ['x_coord', 'y_coord', 'timepoint', 'labels'])
        """
        self.df = df.copy()
        self.identifiers = identifiers
        self.number_of_objects = len(df)
        self.number_of_timepoints = np.max(np.unique(df[identifiers[2]]))
        self.adjacency_matrix = np.zeros([self.number_of_objects,
                                          self.number_of_objects])
        # add unique identifiers to df
        self.df['unique_id'] = list(range(0, self.number_of_objects))
        # link timepoints to get segments
        self.__get_track_segments()
        # try to link the segments among themselves
        print('linking track segments across timepoints')
        self.__close_gaps()
        # get the final tracks
        self.G2 = nx.DiGraph(self.adjacency_matrix)
        self.number_of_tracks = nx.number_weakly_connected_components(self.G2)
        self.tracks = [sorted(c) for c in sorted(
            nx.weakly_connected_components(self.G2),
            key=len,
            reverse=True)]

        self.tracks_by_label = [list(self.df[self.identifiers[3]].iloc[
            sorted(c)]) for c in sorted(
                nx.weakly_connected_components(self.G2),
                key=len,
                reverse=True)]

        # add column for track ids
        track_ids = []
        for obj in range(0, self.number_of_objects):
            for track_id, track in enumerate(self.tracks):
                if obj in track:
                    track_ids.append(track_id)

        self.df['track_id'] = track_ids

    def __switch_labels(self):
        relabeled_movie = np.zeros(np.shape(self.movie), dtype='uint16')
        bar = Bar('switching labels', max=self.number_of_timepoints,
                  check_tty=False, hide_cursor=False)
        for t in range(0, self.number_of_timepoints):
            label_image = self.movie[t, :, :]
            old_labels = self.df['label'].loc[
                self.df.timepoint == t]
            new_labels = self.df['track_id'].loc[
                self.df.timepoint == t]
            arr = np.zeros(label_image.max() + 1, dtype='uint16')
            arr[old_labels] = new_labels
            relabeled_movie[t, :, :] = arr[label_image]
            bar.next()
        bar.finish()

        return relabeled_movie

    def track_label_images(self, movie):
        '''
        Tracks objects in label images over timepoints

        Takes a 3D numpy array (t, x, y) with labelled objects and tries
        to link them between timepoints.

        Parameters
        ----------
        movie: np.array
            3D numpy array of labelled objects (t, x, y)

        Returns
        -------
        tracked_movie: np.array
            3D numpy array with objects relabelled as their track id
        df: pd.DataFrame
            feature measurements used for tracking
        '''

        self.movie = movie
        self.number_of_timepoints = np.size(self.movie, 0)
        df = pd.DataFrame()
        # measure centroids of objects at all timepoints
        bar = Bar('measuring centroids', max=self.number_of_timepoints,
                  check_tty=False, hide_cursor=False)
        for t in range(0, self.number_of_timepoints):
            current_features = regionprops_table(self.movie[t, :, :],
                                                 properties=['label',
                                                             'centroid'])
            current_features['timepoint'] = t
            current_features = pd.DataFrame(current_features)
            df = df.append(current_features)
            bar.next()
        bar.finish()
        # track the objects in the df
        self.track_df(df,
                      ['centroid-0', 'centroid-1', 'timepoint', 'label'])
        # relabel the movie according to the track id
        self.relabeled_movie = self.__switch_labels()
