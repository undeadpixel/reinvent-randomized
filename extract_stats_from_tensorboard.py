#!/usr/bin/env python
#  coding=utf-8

"""
Collects stats for an existing RNN model.
"""

import argparse
import os
import glob
import collections

import tensorboard.backend.event_processing.event_accumulator as tbea

import utils.log as ul


class ExtractStatsFromTensorboardRunner:
    """Samples an existing RNN model."""

    def __init__(self, stats_folder_path, output_folder_path):
        """
        Creates a CollectStatsFromModelRunner.
        :param model_path: The input model path.
        :return:
        """
        self._stats_folder_path = stats_folder_path
        self._output_folder_path = output_folder_path
        self.data = None

    def run(self):
        """
        Extracts the stats in CSV format.
        """
        os.makedirs(self._output_folder_path, exist_ok=True)
        self.data = collections.defaultdict(list)
        self._process_folder(self._stats_folder_path, prefix=[])
        for tag in self.data.keys():
            sorted_data = sorted(self.data[tag], key=lambda x: x[1])
            with open("{}/{}.csv".format(self._output_folder_path, tag), "w+") as csv_file:
                for wall_time, step, value in sorted_data:
                    csv_file.write("{:.0f},{:d},{:.10f}\n".format(wall_time, step, value))

    def _process_folder(self, path, prefix):
        LOG.info("Processing folder %s", "/".join(prefix))
        paths = [os.path.normpath(p) for p in glob.glob("{}/*".format(path))]
        for file_path in filter(os.path.isfile, paths):
            self._process_file(file_path, prefix)

        for dir_path in filter(os.path.isdir, paths):
            dirname = os.path.basename(dir_path)
            self._process_folder(dir_path, prefix + [dirname])

    def _process_file(self, path, prefix):
        event_acc = tbea.EventAccumulator(path).Reload()
        scalar_tags = event_acc.Tags()["scalars"]
        for tag in scalar_tags:
            if prefix:
                if len(scalar_tags) == 1:
                    tag_name = ".".join(prefix)
                else:
                    tag_name = "{}.{}".format(".".join(prefix), tag.replace("/", "."))
            else:
                tag_name = tag.replace("/", ".")
            for scalar_val in event_acc.Scalars(tag):
                self.data[tag_name].append(list(scalar_val))


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(
        description="Extracts the stats in CSV format from the Tensorboard output. \
            It extracts the csv values to CSV files, each file with the same name as the tag and a folder structure equivalent as the one found inside.")
    parser.add_argument("--stats-folder-path", "-s",
                        help="Path to the main directory where the stats are.", type=str, required=True)
    parser.add_argument("--output-folder-path", "-o",
                        help="Path to a folder (will be created if it doesn't exist) where all the csv files will be stored.", type=str, required=True)

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def main():
    """Main function."""
    args = parse_args()
    runner = ExtractStatsFromTensorboardRunner(**args)
    runner.run()


LOG = ul.get_logger("extract_stats_from_tensorboard")
if __name__ == "__main__":
    main()
