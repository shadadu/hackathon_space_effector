import os
import faery

def run(aedat4_file, output_mp4_file):

    faery.events_stream_from_file(aedat4_file) \
        .regularize(frequency_hz=60.0) \
        .render(decay="exponential", tau="00:00:00.200000", colormap=faery.colormaps.starry_night) \
        .to_file(output_mp4_file)

if __name__ == "__main__":
    # Sample file from EVOS dataset: https://carleton.ca/spacecraft/datasets/
    work_dir = os.getcwd()
    aedat4_file = work_dir + '/DGM_Robotics_in_Microgravity_Environments/' + 'FRDR_dataset_1533_download_970_202602231442' + '/CC-T-DARK/recording_20251029_140939.aedat4'
    output_mp4 = work_dir + '/DGM_Robotics_in_Microgravity_Environments/' + 'FRDR_dataset_1533_download_970_202602231442' + '/CC-T-DARK/' + 'recording_20251029_140939.mp4'
    run(aedat4_file, output_mp4)
