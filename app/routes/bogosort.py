"""
bogosort.py - interactive sorting-demo routes for the Flask app

This module provides a visual demonstration of Bogosort and MergeSort using
Flask routes backed by asynchronous background workers.

Sorting execution is separated from request handling so expensive animations do
not block the web server thread. This keeps the UI responsive while long-running
sorts execute independently.

The module intentionally contrasts Bogosort against MergeSort to demonstrate
algorithmic complexity differences visually rather than only theoretically.
It also reveals a time processing difference.
"""

from flask import Blueprint, redirect, render_template, url_for, request, send_file, abort
import threading
import os
import time
import logging
from app.services.sorting_service import SortingService

logger = logging.getLogger(__name__)

bogosort_demo = Blueprint('bogosort', __name__, url_prefix='/sort-demo')

_sorting_state = {
    'state': None,
    'final_iteration': 0,
    'sorted': False,
    'error': None,
    'stop_flag': {'stop': False},
    'algorithm': 'bogosort',
    'seed': '',
}
_sorting_thread = None


def _reset_state():
    global _sorting_state, _sorting_thread
    _sorting_state = {
        'state': None,
        'final_iteration': 0,
        'sorted': False,
        'error': None,
        'stop_flag': {'stop': False},
        'algorithm': 'bogosort',
        'seed': '',
    }
    _sorting_thread = None


@bogosort_demo.route('/', methods=['GET', 'POST'])
def bogosort():
    if request.method == 'POST':
        return handle_post()
    return handle_get()


def handle_post():
    global _sorting_state, _sorting_thread

    algorithm = request.form.get('algorithm', 'bogosort')
    seed_str = request.form.get('seed', '').strip()
    try:
        seed = int(seed_str) if seed_str else None
    except ValueError:
        logger.warning(f"Invalid seed value: {seed_str}, using random seed")
        seed = None

    if _sorting_state['state'] != 'running':
        try:
            words, counts = SortingService.load_shuffled_toxic_words(seed=seed)
            SortingService.save_distribution_plot(words, counts, '/tmp/word_distribution.png')

            stop_flag = {'stop': False}
            _sorting_state['state'] = 'running'
            _sorting_state['algorithm'] = algorithm
            _sorting_state['seed'] = seed_str
            _sorting_state['stop_flag'] = stop_flag

            target = background_mergesort if algorithm == 'mergesort' else background_bogosort
            gif = '/tmp/mergesort_sorting.gif' if algorithm == 'mergesort' else '/tmp/bogosort_sorting.gif'
            _sorting_thread = threading.Thread(
                target=target,
                args=(words, counts, gif, stop_flag),
                daemon=True
            )
            _sorting_thread.start()

        except Exception as e:
            _sorting_state['state'] = 'error'
            _sorting_state['error'] = str(e)

    return redirect(url_for('bogosort.bogosort'))


def handle_get():
    state = _sorting_state['state']
    algorithm = _sorting_state['algorithm']
    seed = _sorting_state['seed']

    dist_url = url_for('bogosort.serve_media', filename='word_distribution.png') + f'?v={int(time.time())}'
    gif_name = 'mergesort_sorting.gif' if algorithm == 'mergesort' else 'bogosort_sorting.gif'
    gif_url = url_for('bogosort.serve_media', filename=gif_name) + f'?v={int(time.time())}'

    if state == 'running':
        return render_template(
            'sort-demo.html',
            dist_url=dist_url, gif_url=gif_url,
            show_form=False, show_spinner=True, show_gif=False,
            algorithm=algorithm, seed=seed
        )

    if state == 'done':
        return render_template(
            'sort-demo.html',
            dist_url=dist_url, gif_url=gif_url,
            show_form=False, show_spinner=False, show_gif=True,
            sorted=_sorting_state['sorted'],
            iteration=_sorting_state['final_iteration'],
            algorithm=algorithm, seed=seed
        )

    if state == 'error':
        return render_template(
            'sort-demo.html',
            dist_url=dist_url, gif_url=gif_url,
            show_form=True, show_spinner=False, show_gif=False,
            error=_sorting_state['error'],
            algorithm='bogosort', seed=''
        )

    # Initial state — generate the distribution plot
    try:
        words, counts = SortingService.load_shuffled_toxic_words(seed=None)
        SortingService.save_distribution_plot(words, counts, '/tmp/word_distribution.png')
    except Exception as e:
        return render_template(
            'sort-demo.html',
            dist_url=dist_url, gif_url=gif_url,
            show_form=True, show_spinner=False, show_gif=False,
            error=f'Failed to load data: {str(e)}',
            algorithm='bogosort', seed=''
        )

    return render_template(
        'sort-demo.html',
        dist_url=dist_url, gif_url=gif_url,
        show_form=True, show_spinner=False, show_gif=False,
        algorithm='bogosort', seed=''
    )


@bogosort_demo.route('/media/<filename>')
def serve_media(filename):
    allowed = {'word_distribution.png', 'bogosort_sorting.gif', 'mergesort_sorting.gif'}
    if filename not in allowed:
        abort(404)
    path = os.path.join('/tmp', filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path)


@bogosort_demo.route('/stop', methods=['POST'])
def stop_sorting():
    _sorting_state['stop_flag']['stop'] = True
    return redirect(url_for('bogosort.bogosort'))


@bogosort_demo.route('/reset', methods=['GET'])
def reset():
    _reset_state()
    return redirect(url_for('bogosort.bogosort'))


def background_bogosort(words, counts, gif_filename, stop_flag):
    try:
        snapshots = SortingService.bogosort_snapshots(
            words, counts, max_iterations=1000, stop_flag=stop_flag
        )
        SortingService.save_sort_animation(snapshots, gif_filename, title='Bogosort Animation', stop_flag=stop_flag)
        if stop_flag['stop']:
            _sorting_state['state'] = None
            return
        _sorting_state['state'] = 'done'
        _sorting_state['final_iteration'] = snapshots[-1][1] if snapshots else 0
        _sorting_state['sorted'] = SortingService.is_sorted([x[1] for x in snapshots[-1][0]]) if snapshots else False
    except Exception as e:
        _sorting_state['state'] = 'error'
        _sorting_state['error'] = str(e)


def background_mergesort(words, counts, gif_filename, stop_flag):
    try:
        snapshots = SortingService.mergesort_snapshots(
            words, counts, stop_flag=stop_flag
        )
        SortingService.save_sort_animation(snapshots, gif_filename, title='MergeSort Animation', stop_flag=stop_flag)
        if stop_flag['stop']:
            _sorting_state['state'] = None
            return
        _sorting_state['state'] = 'done'
        _sorting_state['final_iteration'] = snapshots[-1][1] if snapshots else 0
        _sorting_state['sorted'] = SortingService.is_sorted([x[1] for x in snapshots[-1][0]]) if snapshots else False
    except Exception as e:
        _sorting_state['state'] = 'error'
        _sorting_state['error'] = str(e)
