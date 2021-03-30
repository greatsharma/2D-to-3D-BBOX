from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

input_path = "inputs/hyderabad_clip7.avi"
output_path = "inputs/hyderabad_clip8.avi"
start_time = 1805
end_time = 1815

ffmpeg_extract_subclip(input_path, start_time, end_time, targetname=output_path)

# fmpeg -ss 00:07:25 -i inputs/hyderabad_clip2.avi -t 00:00:05 -codec copy inputs/hyderabad_clip10.avi

"""
:: Create File List
echo file file1.mp4 >  mylist.txt 
echo file file2.mp4 >> mylist.txt
echo file file3.mp4 >> mylist.txt

:: Concatenate Files
ffmpeg -f concat -i mylist.txt -c copy output.mp4
"""
