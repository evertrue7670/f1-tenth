import os
import cv2
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import argparse

'''
    Video function
'''
def create_video(frames, fps=120, output_name='output'):
    _output_name = output_name + '_.mp4'
    out = cv2.VideoWriter(_output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
    os.system(f'ffmpeg -i {_output_name} -vcodec libx264 {output_name + ".mp4"} -hide_banner -loglevel error')      # convert to h264
    os.system(f'rm -rf {_output_name}')

def video_clip(video_path, start_time, end_time):
    ffmpeg_extract_subclip(f"{video_path}.mp4", start_time, end_time, targetname=f"{video_path}_clip.mp4")

def video_to_gif(video_path, name=None, speed=1):
    name = name or video_path
    clip = VideoFileClip(f'{video_path}.mp4')
    clip.speedx(speed).write_gif(f'{name}.gif')

'''
    Merge function
'''
# for concat videos,
### mylist: 
###        file 'seed7_1_clip.mp4'
###        file 'seed7_2_clip.mp4'
###        file 'seed7_3_clip.mp4'
### in terminal:
###        ffmpeg -f concat -i mylist.txt -c copy output.mp4

def horizontally_merge_video(video_1, video_2, output_name='output'):
    os.system(f'ffmpeg -i {video_1}.mp4 -i {video_2}.mp4  -filter_complex hstack {output_name}.mp4')