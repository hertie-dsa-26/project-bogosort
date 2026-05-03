from flask import Blueprint, redirect, render_template, url_for, request
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import io
import random
import time
import threading

bogosort_demo = Blueprint('bogosort', __name__,url_prefix="/bogosort")

sorting_status = {"state": None, "final_iteration": 0, "sorted": False}

@bogosort_demo.route('/', methods=['GET', 'POST'])
def bogosort():
    
    static_gif = 'app/static/bogosort_sorting.gif'
    static_dist = 'app/static/word_distribution.png'
    # Load and shuffle the .npy
    if sorting_status["state"] in (None, "done", "error"):
        words, counts = load_shuffled_toxic_words()
        save_distribution_plot(words, counts, static_dist)
        # Save for background sort if needed
        sorting_status["words"] = words
        sorting_status["counts"] = counts

    # Generate bogosort snapshots
    #snapshots = bogosort_snapshots(words, counts, max_iterations=500)
    #gif_path = 'app/static/bogosort_sorting.gif'
    gif_url = url_for('static', filename='bogosort_sorting.gif') + f'?v={int(time.time())}'
    dist_url = url_for('static', filename='word_distribution.png') + f'?v={int(time.time())}'
    
    iter_count = sorting_status.get("final_iteration", 0)
    sorted_flag = sorting_status.get("sorted", False)

    if request.method == 'POST':
        # Start sorting in a thread
        if sorting_status["state"] != "running":
            sorting_status["state"] = "running"
            thread = threading.Thread(target=background_bogosort, args=(words, counts, static_gif, sorting_status))
            thread.start()
        return redirect(url_for('bogosort.bogosort'))
    
    if sorting_status["state"] == "running":
        # Sorting in progress
        return render_template(
            'bogosort.html',
            dist_url=dist_url,
            gif_url=gif_url,
            show_gif=False,
            show_dist=False,
            show_spinner=True,
            sorted=sorted_flag,
            iteration=iter_count
        )
    
    elif sorting_status["state"] == "done":
        return render_template(
            'bogosort.html',
            dist_url=dist_url,
            gif_url=gif_url,
            show_gif=True,
            show_dist=False,
            show_spinner=False,
            sorted=sorting_status["sorted"],
            iteration=sorting_status["final_iteration"]
        )
    
    else:
        # Idle state, show distribution
        return render_template(
            'bogosort.html',
            dist_url=dist_url,
            gif_url=gif_url,
            show_dist=True,
            show_gif=False,
            show_spinner=False,
            sorted=sorted_flag,
            iteration=iter_count
        )
 
# -- Helper Functions --

def background_bogosort(words, counts, gif_filename, status_flag):
    status_flag['state'] = 'running'
    try:
        snapshots = bogosort_snapshots(words, counts, max_iterations=1000)
        save_sort_animation(snapshots, gif_filename, title='Bogosort Animation')
        status_flag['state'] = 'done'
        # Mark sorting/capped in session if desired
        status_flag['final_iteration'] = snapshots[-1][1]
        status_flag['sorted'] = is_sorted([x[1] for x in snapshots[-1][0]])
    except Exception as e:
        status_flag['state'] = 'error'
        status_flag['err'] = str(e)

def load_shuffled_toxic_words(filename='./data/top_toxic_words.npy'):
    arr = np.load(filename, allow_pickle=True)
    lst = list(arr)
    random.shuffle(lst)
    # Unzipping to get parallel word and count lists
    words, counts = zip(*lst)
    counts = np.array(counts).astype(int)
    return list(words), list(counts)

def is_sorted(counts):
    return all(counts[i] >= counts[i+1] for i in range(len(counts)-1))

def save_distribution_plot(words, counts, filename='static/word_distribution.png'):
    plt.figure(figsize=(8,4))
    bars = plt.bar(words, counts, color='deepskyblue')
    plt.title("Toxic Word Count Distribution")
    plt.xlabel("Word")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(filename)
    plt.close()

def bogosort_snapshots(words, counts, max_iterations=500):
    arr = list(zip(words, counts))
    snapshots = []
    temp = arr[:]
    iterations = 0
    while not is_sorted([x[1] for x in temp]):
        random.shuffle(temp)
        snapshots.append((list(temp), iterations))
        iterations += 1
        if iterations > max_iterations:
            break
    snapshots.append((list(temp), iterations))
    return snapshots

def save_sort_animation(snapshots, filename='static/bogosort_sorting.gif', title='Bogosort Animation'):
    fig, ax = plt.subplots(figsize=(6, 4))
    frames = []
    for idx, (snap, iter_no) in enumerate(snapshots):
        ax.clear()
        tmp_words, tmp_counts = zip(*snap)
        bars = ax.bar(tmp_words, tmp_counts, color='orchid')
        ax.set_ylim(0, max(tmp_counts) + 2)
        is_final = (idx == len(snapshots)-1)
        sorted_flag = is_sorted([x[1] for x in snap])
        title_text = f"{title} (Iteration: {iter_no})"
        if is_final:
            title_text += " - Sorted!" if sorted_flag else " - NOT sorted!"
        ax.set_title(title_text)
        #ax.set_title(f"{title} (Iteration: {iter_no})")
        #ax.set_title(title)
        ax.set_ylabel("Count")
        ax.set_xlabel("Word")
        ax.set_xticklabels(tmp_words, rotation=45, ha="right")
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.v3.imread(buf))
        buf.close()
    imageio.mimsave(filename, frames, duration=1)
    plt.close(fig)