from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement import CameraMovementEstimator
from perspective_transformer import PerspectiveTransformer
from distance_and_speed_estimator import DistanceAndSpeedEstimator

def main():
    # Read vid
    frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Add  Positions to Tracks
    tracker.add_positions_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.estimate_camera_movement(frames,read_from_stub=True,stub_path='stubs/camera_movement_stubs.pkl')

    # Add Adjusted Positions to Tracks
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # Perspective Transformer
    perspective_transformer = PerspectiveTransformer()
    perspective_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball
    tracks['ball'] = tracker.interpolate_ball(tracks['ball']) 

    # Distance and Speed Estimator
    distance_and_speed_estimator = DistanceAndSpeedEstimator()
    distance_and_speed_estimator.add_distance_and_speed_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()

    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Possesion
    player_ball_assigner = PlayerBallAssigner()
    team_possession = []
    for  frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_possession.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_possession.append(team_possession[-1])

    
    team_possession = np.array(team_possession)

    # Draw output
    ## Draw Tracks
    output_frames = tracker.draw_annotations(frames,tracks,team_possession)

    ## Draw Camera Movement
    output_frames = camera_movement_estimator.draw_camera_movement(output_frames,camera_movement_per_frame)

    ## Draw Distance and Speed
    output_frames = distance_and_speed_estimator.draw_distance_and_speed(output_frames,tracks)

    # Save vid
    save_video(output_frames,'output_videos/output_video.avi')


if __name__ == '__main__':
    main()