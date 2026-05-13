import numpy as np
import random
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import io

logger = logging.getLogger(__name__)


class SortingService:
    
    @staticmethod
    def load_shuffled_toxic_words(filename='./app/data/top_toxic_words.npy', seed=None, top_n=20):
        try:
            arr = np.load(filename, allow_pickle=True)
        except FileNotFoundError as exc:
            logger.error(f"Toxic words file not found: {filename}", exc_info=True)
            raise FileNotFoundError(f"Toxic words data file missing at {filename}") from exc
        except (OSError, ValueError) as exc:
            logger.error(f"Error loading toxic words: {exc}", exc_info=True)
            raise ValueError(f"Corrupted toxic words file: {exc}") from exc

        try:
            lst = list(arr)
            lst.sort(key=lambda item: item[1], reverse=True)
            lst = lst[:top_n]
            rng = random.Random(seed)
            rng.shuffle(lst)
            if not lst:
                return [], []
            words, counts = zip(*lst)
            counts = np.array(counts).astype(int)
            return list(words), list(counts)
        except (TypeError, ValueError, IndexError) as exc:
            logger.error(f"Invalid format in toxic words array: {exc}", exc_info=True)
            raise ValueError(f"Malformed toxic words data: {exc}") from exc

    @staticmethod
    def is_sorted(counts):
        return all(counts[i] >= counts[i+1] for i in range(len(counts)-1))

    @staticmethod
    def save_distribution_plot(words, counts, filename='static/word_distribution.png'):
        try:
            plt.figure(figsize=(8, 4))
            plt.bar(words, counts, color='deepskyblue')
            plt.title("Toxic Word Count Distribution")
            plt.xlabel("Word")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(rect=[0, 0.15, 1, 1])
            plt.savefig(filename)
            plt.close()
        except (OSError, PermissionError, IOError) as exc:
            logger.error(f"Failed to save distribution plot to {filename}: {exc}")
            plt.close()
            raise IOError(f"Failed to generate plot: {exc}") from exc
        except Exception as exc:
            logger.error(f"Unexpected error saving plot: {exc}", exc_info=True)
            plt.close()
            raise

    @staticmethod
    def bogosort_snapshots(words, counts, max_iterations=500, seed=None, stop_flag=None):
        arr = list(zip(words, counts))
        snapshots = []
        temp = arr[:]
        iterations = 0
        rng = random.Random(seed)
        while not SortingService.is_sorted([x[1] for x in temp]):
            if stop_flag and stop_flag.get('stop', False):
                break
            rng.shuffle(temp)
            snapshots.append((list(temp), iterations))
            iterations += 1
            if iterations % 100 == 0:
                logger.debug(f"Bogosort iteration {iterations}, still not sorted")
            if iterations > max_iterations:
                break
        snapshots.append((list(temp), iterations))
        return snapshots

    @staticmethod
    def mergesort_snapshots(words, counts, seed=None, stop_flag=None):
        arr = list(zip(words, counts))
        snapshots = []

        def merge_sort(arr, snapshots, stop_flag):
            if len(arr) > 1:
                if stop_flag and stop_flag.get('stop', False):
                    return

                mid = len(arr) // 2
                left_half = arr[:mid]
                right_half = arr[mid:]

                merge_sort(left_half, snapshots, stop_flag)
                merge_sort(right_half, snapshots, stop_flag)

                i = j = k = 0

                while i < len(left_half) and j < len(right_half):
                    if left_half[i][1] >= right_half[j][1]:
                        arr[k] = left_half[i]
                        i += 1
                    else:
                        arr[k] = right_half[j]
                        j += 1
                    k += 1

                while i < len(left_half):
                    arr[k] = left_half[i]
                    i += 1
                    k += 1

                while j < len(right_half):
                    arr[k] = right_half[j]
                    j += 1
                    k += 1

                snapshots.append((list(arr), len(snapshots)))

        merge_sort(arr, snapshots, stop_flag)
        return snapshots

    @staticmethod
    def save_sort_animation(snapshots, filename='static/bogosort_sorting.gif', title='Sorting Animation', stop_flag=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        frames = []
        try:
            for idx, (snap, iter_no) in enumerate(snapshots):
                if stop_flag and stop_flag.get('stop', False):
                    plt.close(fig)
                    return
                try:
                    ax.clear()
                    tmp_words, tmp_counts = zip(*snap)
                    ax.bar(tmp_words, tmp_counts, color='orchid')
                    ax.set_ylim(0, max(tmp_counts) + 2)
                    is_final = (idx == len(snapshots) - 1)
                    sorted_flag = SortingService.is_sorted([x[1] for x in snap])
                    title_text = f"{title} (Iteration: {iter_no})"
                    if is_final:
                        title_text += " - Sorted!" if sorted_flag else " - NOT sorted!"
                    ax.set_title(title_text)
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Word")
                    ax.set_xticklabels(tmp_words, rotation=45, ha="right")
                    fig.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    frames.append(imageio.v3.imread(buf))
                    buf.close()
                    logger.debug(f"Generated animation frame {idx + 1}/{len(snapshots)}")
                except (TypeError, ValueError) as exc:
                    logger.error(f"Error processing snapshot {idx}: {exc}")
                    continue

            imageio.mimsave(filename, frames, duration=1)
            logger.info(f"Successfully saved animation to {filename}")
            plt.close(fig)
        except (OSError, PermissionError, IOError) as exc:
            logger.error(f"Failed to save animation to {filename}: {exc}")
            plt.close(fig)
            raise IOError(f"Failed to save animation: {exc}") from exc
        except Exception as exc:
            logger.error(f"Unexpected error during animation generation: {exc}", exc_info=True)
            plt.close(fig)
            raise