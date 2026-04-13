###################################################################################
#   Copyright (c) 2021, 2024 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################

# Edited by Gustavo Vilar de Farias (Politecnico di Torino) - 2026
# Added fault injection capabilities

"""
Typical ai_runner example
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import argparse
import logging
import re
import multiprocessing
import pathlib

from time import perf_counter

import numpy as np

try:
    from stm_ai_runner import AiRunner, __version__
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from stm_ai_runner import AiRunner, __version__
    except ImportError as e:
        print('ModuleNotFoundError:', e,  # noqa: T201
              '- Update the \'PYTHONPATH\' to find the stm_ai_runner Python module.')
        sys.exit(1)


_DEFAULT = 'serial'
__VERSION__ = "1.3.1"


# History
#
#  1.3.0 - disable the dump of the intermediate tfl tensors
#          by default, "--with-data" should be used. User should be aware that
#          for some TFlite models, when "experimental_preserve_all_tensors" option is used,
#          outputs of the model can be different.
#  1.3.1 - add check/detection to import stm_ai_runner module w/o PYTHONPATH
#


def _shape_desc(values):
    values = list(values)
    return '(' + ','.join([str(v) for v in values]) + ')'


def topk_by_sort(values, k, axis=None, ascending=True):
    """Return topK values - (index[], values[])"""  # noqa: DAR101,DAR201,DAR40
    if not ascending:
        values *= -1
    ind = np.argsort(values, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)
    if not ascending:
        values *= -1
    val = np.take_along_axis(values, ind, axis=axis)
    return ind, val


def print_table(table, print_fn=None, with_header=True, indent=0):
    """Simple utility function to print a formatted table"""  # noqa: DAR101,DAR201,DAR401

    print_drv = print if print_fn is None else print_fn  # noqa: T202

    def p_fn(str_):
        """."""  # noqa: DAR101,DAR201,DAR401
        print_drv('{}{}'.format(' ' * indent, str_))

    if 'rows' not in table:
        return

    if 'fmt' not in table:
        fmt = [''] * len(table['rows'][0])
    else:
        fmt = table['fmt']

    row_size = [0] * len(table['rows'][0])
    for row in table['rows']:
        c_size = []
        if not row:
            continue
        for item in row:
            c_size.append(len(str(item)) + 2)
        row_size = [max(r, cur) for r, cur in zip(row_size, c_size)]

    line_size = sum(row_size) + len(row_size) - 1

    if 'title' in table:
        p_fn('{}'.format(table['title']))
        line_size = max(line_size, len(table['title']))

    p_fn('-' * line_size)

    for idx, row in enumerate(table['rows']):
        if not row:
            p_fn('-' * line_size)
            continue
        str_ = ''
        for size_, fmt_, item in zip(row_size, fmt, row):
            item = str(item)
            if '>' in fmt_:
                str_ += ' ' * (size_ - len(item)) + item
            else:
                str_ += item + ' ' * (size_ - len(item))
            str_ += ' '
        p_fn(str_)
        if idx == 0 and with_header:
            p_fn('-' * line_size)
    if with_header:
        p_fn('-' * line_size)

def load_h5_model(h5_file):
    import tensorflow as tf  # ignore_missing_imports: ignore

    if not h5_file:
        return None

    model = tf.keras.models.load_model(h5_file, compile=False)

    return model

def invoke_h5_model(model, inputs):
    if model is None:
        return []

    outputs = model.predict(inputs)

    return outputs

def check_io(runner, tfl_interpreter, logger):
    """."""  # noqa:DAR101,DAR201,DAR401

    if tfl_interpreter is None or runner is None:
        return

    logger.info('Checking IO tensors...')

    tfl_is = tfl_interpreter.get_input_details()
    tfl_os = tfl_interpreter.get_output_details()
    ai_is = runner.get_input_infos()
    ai_os = runner.get_output_infos()

    if len(tfl_is) != len(ai_is) or len(tfl_os) != len(ai_os):
        raise ValueError('IO size are not consistent')

    for tf_i, ai_i in zip(tfl_is, ai_is):
        if tf_i['dtype'] != ai_i['type']:
            raise ValueError(f"Invalid I dtype, {ai_i['type']}, expected {tf_i['dtype']}")
        if np.prod(tf_i['shape'][1:]) != np.prod(ai_i['shape'][1:]):
            raise ValueError(f"Invalid I shape, {ai_i['shape']}, expected {tf_i['shape']}")

    for tf_o, ai_o in zip(tfl_os, ai_os):
        if tf_o['dtype'] != ai_o['type']:
            raise ValueError(f"Invalid O dtype, {ai_o['type']}, expected {tf_o['dtype']}")
        if np.prod(tf_o['shape'][1:]) != np.prod(ai_o['shape'][1:]):
            raise ValueError('Invalid O shape')

    logger.info(' OK')


def gen_datas(runner, name, batch, range_value, logger, rnd_gen):
    """."""  # noqa:DAR101,DAR201

    logger.info('')
    logger.info('Generating input data... (b={}, range={})'.format(batch, range_value))

    val = range_value
    if range_value and len(range_value) == 1:
        val = [range_value[0]]
        logger.info(' same value is used for all items: {}'.format(val))

    inputs = runner.generate_rnd_inputs(name, batch_size=batch, rng=rnd_gen, val=val)

    logger.info('')

    return inputs


def remove_nans(lhs, rhs):
    """Remove NaNs and infinities from a pair of arrays."""  # noqa: DAR101,DAR201
    mask = (np.isnan(lhs) & np.isnan(rhs)) | ((lhs == np.inf) & (rhs == np.inf)) | ((lhs == -np.inf) & (rhs == -np.inf))
    lhs[mask] = rhs[mask] = 0.0

    return lhs, rhs


def fix_and_check_valid_tag(tag, p_fn):
    """Fix the expected tag definition"""  # noqa:DAR101,DAR201
    tag = tag.upper()
    stag = tag.split('.')
    if len(stag) < 3:
        tag += '.0'

    if bool(re.match(r'E.[0-9]+.[0-9]+', tag)):
        return tag

    if not bool(re.match(r'N.[0-9]+.[0-9]+', tag)):
        p_fn('WARN: \'{}\' invalid tag format'.format(tag))
        return None

    return tag


def get_from_tflm(tag, tfl_internals):
    """Extract a given tensor"""  # noqa:DAR101,DAR201

    if not tfl_internals:
        return None

    if len(tag.split('.')) == 2:
        tag += '.0'

    for internal in tfl_internals:
        for _, port in enumerate(internal['outputs']):
            if tag == port['tag']:
                return port['data']

    return None


def get_from_stm_ai(tag, ai_profiler, ref, p_fn):
    """Extract a given tensor"""  # noqa:DAR101,DAR201

    if not ai_profiler:
        return None

    if len(tag.split('.')) == 2:
        tag += '.0'

    if ai_profiler['c_nodes']:
        for _, c_node in enumerate(ai_profiler['c_nodes']):
            for _, (data, io_tensor) in enumerate(zip(c_node['data'], c_node['io_tensors'])):
                if data.size:
                    if io_tensor.tag == tag:
                        if ref is not None:
                            if ref.dtype == data.dtype and ref.size == data.size:
                                return data
                            else:
                                msg_ = '{} tag, type: {} -> {}, size: {} -> {}'.format(tag, data.dtype,
                                                                                       ref.dtype,
                                                                                       data.size, ref.size)
                                p_fn('ERR: Incompatible tensor fmt for STM.AI.' + msg_)

    if not ai_profiler['c_nodes']:
        p_fn('WARN: STAM.AI model should be executed not only with IO_ONLY option')

    return None


def get_histogram(ref, desc, density=False):
    """Return histogram of the error"""  # noqa:DAR101,DAR201

    import termplotlib as tpl

    res = '\n'

    counts, bin_edges = np.histogram(ref, bins=64, range=(-128, 127), density=density)
    bin_ = ['  {:-4f}'.format(v) for v in bin_edges]
    fig = tpl.figure()
    fig.barh(counts, bin_, force_ascii=True)
    res += f' {desc} distribution\n'
    res += ' ' + '-' * 50 + '\n'
    res += fig.get_string()
    res += '\n' + ' ' + '-' * 50 + '\n'
    return res


def compute_metrics(ai_profiler, quant_params, ai_outputs, tfl_internals, tfl_outputs, mapper, histo, logger):
    """."""  # noqa:DAR101,DAR201
    if not tfl_outputs:
        logger.warning(" WARN: Skipped, no TFL data...")
        return

    indent = 1
    print_drv = print if logger is None else logger.info  # noqa: T202

    def p_fn(str_):
        """."""  # noqa:DAR101
        print_drv('{}{}'.format(' ' * indent, str_))

    def dequant(idx, vals, quant_params):
        """Dequant the quantized values"""  # noqa: DAR201,DAR101
        if not quant_params:
            return vals
        param = quant_params[idx]
        if len(param['scales']):
            return (vals.astype(np.float32) - param['zero_points'][0]) * param['scales'][0]
        return vals

    def _bias(ref, pred):
        """Return bias"""  # noqa: DAR201,DAR101
        return np.mean(ref - pred)

    def _min(ref, pred):
        """Return min diff"""  # noqa: DAR201,DAR101
        return (ref - pred).min()

    def _max(ref, pred):
        """Return max diff"""  # noqa: DAR201,DAR101
        return (ref - pred).max()

    def _std(ref, pred):
        """Return bias"""  # noqa: DAR201,DAR101
        return np.std(ref - pred, ddof=1)

    def _mse(ref, pred):
        """Return Mean Squared Error (MSE)"""  # noqa: DAR201,DAR101
        return np.mean((ref - pred) ** 2)

    def _rmse(ref, pred):
        """Return Root Mean Squared Error (RMSE)."""  # noqa: DAR201,DAR101
        return np.sqrt(_mse(ref, pred))

    def _cosine(ref, pred):
        """Return Cosine Similarity (COS)"""  # noqa: DAR201,DAR101
        ref_ = ref
        pred_ = pred
        numerator = np.dot(ref_.flatten(), pred_.flatten())
        denominator = np.linalg.norm(ref_.flatten()) * np.linalg.norm(pred_.flatten() + np.finfo(np.float32).eps)
        if np.isclose(numerator, denominator, rtol=np.finfo(np.float32).eps):
            err = np.float32(1.0)
        else:
            err = np.float32(numerator / denominator)
        return err

    def _nse(ref, pred):
        """Return Nash-Sutcliffe efficiency criteria"""  # noqa: DAR201,DAR101
        # Normalized statistic that determines the relative magnitude of
        # the residual variance ("noise") compared to the measured data
        # variance ("information").
        # Regressor model - range = (-inf, 1], bigger is best
        #   numerator = np.sum((ref.flatten() - pred.flatten()) ** 2)
        #   denominator = np.sum((ref.flatten() - np.mean(ref.flatten())) ** 2)
        #   return 1 - numerator / denominator
        nse_ = 1 - _mse(ref, pred) / ((np.std(ref, ddof=1) ** 2) + np.finfo(np.float32).eps)
        return nse_

    def _histogram(ref, pred, histo=None):
        """Return histogram of the error"""  # noqa:DAR101,DAR201

        import termplotlib as tpl

        diff_all = np.subtract(ref, pred).flatten()
        diff = diff_all[diff_all != 0]

        perc = 1 - ((diff_all.size - diff.size) / diff_all.size)

        no_diff = False
        if diff.size == 0:
            no_diff = True
            diff = diff_all

        if ref.dtype != np.float32:
            min_, max_ = int(np.min(diff)), int(np.max(diff))
        else:
            min_, max_ = np.min(diff), np.max(diff)
        n_bins = 20
        if ((max_ - min_) / n_bins) < 1:
            n_bins = max_ - min_

        if n_bins < 3:
            n_bins = 4
            h_min_, h_max_ = -1, 3
        else:
            h_min_, h_max_ = min_, max_

        counts, bin_edges = np.histogram(diff, bins=int(n_bins), range=(h_min_, h_max_))
        bin_ = [f'  {v:-4.1f}' for v in bin_edges]
        fig = tpl.figure()
        fig.barh(counts, bin_, force_ascii=True)
        desc = f' REF[{np.min(ref)}, {np.max(ref)}], PRED[{np.min(pred)}, {np.max(pred)}],'
        if no_diff:
            desc += f' NO DIFF[{min_:.04f}, {max_:.04f}]'
        else:
            desc += f' {diff.size:,d}/{ref.size:,d} != 0 ({perc * 100:.2f}%), DIFF[{min_:.03f}, {max_:.03f}]'

        # fig.show()
        len_ = len(desc)

        res = '-' * len_ + '\n'
        res += desc
        res += '\n ' + '-' * len_ + '\n'

        if histo and histo == 'full':
            res += get_histogram(ref, 'TFL.REF')
            res += '\n'
            res += ' Error distribution\n'
            res += ' ' + '-' * len_ + '\n'

        res += fig.get_string()

        res += '\n\n'
        res += ' COS  : {:.3f}\n'.format(_cosine(ref, pred))
        res += ' NSE  : {:.3f}\n'.format(_nse(ref, pred))
        res += ' RMSE : {:.8f}\n'.format(_rmse(ref, pred))

        return res

    def build_row_metric_table(first, shape_type, ref, pred):
        row = [first]
        row.append(shape_type)
        row.append('{:.3f}'.format(_cosine(ref, pred)))
        row.append('{:.3f}'.format(_nse(ref, pred)))
        row.append('{:.8f}'.format(_min(ref, pred)))
        row.append('{:.8f}'.format(_max(ref, pred)))
        row.append('{:.8f}'.format(_mse(ref, pred)))
        row.append('{:.8f}'.format(_rmse(ref, pred)))
        row.append('{:.8f}'.format(_bias(ref, pred)))
        row.append('{:.8f}'.format(_std(ref, pred)))
        return row

    table = {'title': 'Computed metrics (tfl/stm.ai)', 'rows': [],
             'fmt': ['', '', '>', '>', '>', '>', '>', '>', '>', '>']}
    table['rows'].append(['tensor', 'item', 'cos', 'nse', 'min', 'max', 'mse', 'rmse', 'bias', 'std'])

    for idx, (tf_o, ai_o) in enumerate(zip(tfl_outputs, ai_outputs)):
        shape_type = '{:,}/{}/{}'.format(tf_o.size, tf_o.dtype, '(' + ','.join([str(v_) for v_ in tf_o.shape]) + ')')
        histo_ = histo if (histo and np.issubdtype(tf_o.dtype, np.integer)) else None
        dtf_o = dequant(idx, tf_o.flatten().astype(np.float32), quant_params)
        dai_o = dequant(idx, ai_o.flatten().astype(np.float32), quant_params)
        table['rows'].append(build_row_metric_table(f'O.{idx}/O.{idx}', shape_type, dtf_o, dai_o))

        if histo_:
            p_fn(f'\n O.{idx}/O.{idx} {shape_type} Histogram of the error "{tf_o.dtype}"')
            p_fn(_histogram(dtf_o, dai_o, histo_))

    if mapper:
        for req_map in mapper:
            req_map = req_map.split(":")
            if len(req_map) == 1:
                req_map.append(req_map[0])
            if len(req_map) != 2:
                p_fn('\nwarning: \'{}\' is an invalid map request - \'N.n.p:N.n.p\''.format(
                    ':'.join(req_map)))
                continue
            req_map[0] = fix_and_check_valid_tag(req_map[0], p_fn)
            req_map[1] = fix_and_check_valid_tag(req_map[1], p_fn)
            if not req_map[0] or not req_map[1]:
                continue
            ref = get_from_tflm(req_map[0], tfl_internals)
            pred = get_from_stm_ai(req_map[1], ai_profiler, ref, p_fn)
            if ref is not None and pred is not None:
                shape_type = '{:,}/{}/{}'.format(ref.size, ref.dtype,
                                                 '(' + ','.join([str(v_) for v_ in ref.shape]) + ')')
                histo_ = histo if (histo and np.issubdtype(ref.dtype, np.integer)) else None
                ref, pred = ref.flatten().astype(np.float32), pred.flatten().astype(np.float32)
                table['rows'].append(build_row_metric_table('{}/{}'.format(req_map[0], req_map[1]),
                                                            shape_type, ref, pred))
                if histo_:
                    p_fn('\n {}/{} {} Histogram of the error'.format(req_map[0], req_map[1], shape_type))
                    p_fn(_histogram(ref, pred, histo_))
            else:
                p_fn('WARN: unable to compare the \'TFM.{}\' tensor with the \'STM.AI.{}\' tensor'.format(
                    req_map[0], req_map[1]))

    p_fn('')
    print_table(table, p_fn)
    if quant_params and len(quant_params[0]['scales']) and quant_params[0]['scales'][0]:
        p_fn('Note: dequantized values are used to compute the metrics')

    p_fn('')
    p_fn(' cos  : Cosine Similarity')
    p_fn(' nse  : Nash-Sutcliffe efficiency')
    p_fn(' mse  : Mean Squared Error')
    p_fn(' rmse : Root Mean Squared Error')
    p_fn(' bias : Bias/Mean')
    p_fn(' std  : Standard Deviation')
    p_fn('')


def save_tensors(inputs, ai_profiler, ai_outputs, logger, dir_="./"):
    """Save the tensors in a npz file"""  # noqa:DAR101,DAR201

    # tag is used as key
    dir_ = pathlib.Path(dir_)
    _arr_dict = {}
    f_name = dir_ / 'tensors_ai.npz'

    logger.info(' Saving AI data... {}'.format(f_name))

    for idx, in_ in enumerate(inputs):
        _arr_dict['I.{}'.format(idx)] = in_

    for idx, out_ in enumerate(ai_outputs):
        _arr_dict['O.{}'.format(idx)] = out_

    if ai_profiler and ai_profiler['c_nodes']:
        for _, c_node in enumerate(ai_profiler['c_nodes']):
            for idx, (data, io_tensor) in enumerate(zip(c_node['data'], c_node['io_tensors'])):
                if data.size:
                    key = io_tensor.tag
                    _arr_dict[key] = data

    np.savez(f_name, **_arr_dict)

def save_classifications(classifications, logger, dir_="./"):
    """Save the classifications"""

    dir_ = pathlib.Path(dir_)
    f_name = dir_ / 'classifications.txt'

    logger.info(' Saving classifications... {}'.format(f_name))

    with open(f_name, 'w') as f:
        f.write(str(list(classifications)))


def gen_f_bit_positions(f_bit_range = 16, f_bit_start = 63, f_bit_step = 32):
    f_bit_positions = []
    for start in range(f_bit_start, -1, -f_bit_step):
        f_bit_positions.extend(list(range(start, start - f_bit_range, -1)))
    return f_bit_positions

def calc_acc(ref, pred):
    right_pred = np.sum((ref == pred))
    return f"{(right_pred / ref.size):.2%}"

def run(args, logger):
    """Main processing function"""  # noqa: DAR101,DAR201,DAR401

    logger.info(f'Running ai_checker v{__VERSION__}..')

    inputs = None
    if args.valinput and os.path.isfile(args.valinput):
        logger.info(f'\n Loading data file : {args.valinput}')

        if args.valinput.endswith('.npy'):
            file_ = np.load(args.valinput)
            inputs = [file_]
        elif args.valinput.endswith('.npz'):
            file_ = np.load(args.valinput)
            keys = ['m_inputs', 'x_test']
            inputs = []
            for entry in file_.files:
                if keys[0] in entry or keys[1] in entry:
                    logger.info(f'  select the "{entry}" entry')
                    inputs = [file_[entry]]
            if not inputs:
                inputs = None
                logger.info(f'{keys} is not available.')
        else:
            inputs = None

        if inputs:
            logger.info('  shape={} dtype={} min={} max={}'.format(inputs[0].shape, inputs[0].dtype,
                                                                   np.min(inputs[0]), np.max(inputs[0])))
        else:
            logger.info('File format is not supported.')

    if inputs:
        args.batch = inputs[0].shape[0]
    
    if args.h5_file:
        h5_model = load_h5_model(args.h5_file)
    else:        
        h5_model = None


    logger.info(f'\nOpening st.ai runtime "{args.desc}" (AiRunner v{__version__})..')

    verbosity = args.verbosity
    runner = AiRunner(debug=args.debug, verbosity=verbosity)
    runner.connect(args.desc)

    if not runner.is_connected:
        logger.error('!!! connection to st.ai runtime has failed..')
        logger.error(' {}'.format(runner.get_error()))
        return 1
    
    logger.debug(runner)
    c_name = runner.names[0] if not args.name else args.name
    session = runner.session(c_name)

    if args.list:
        for c_name in runner.names:
            runner.summary(c_name, print_fn=logger.info, indent=1)
        runner.disconnect()
        return 0

    session.summary(print_fn=logger.info, indent=1)

    if args.ihex:
        logger.info('Loading data... ihex={}'.format(args.ihex))
        session.extension(name=c_name, cmd='write', ihex=args.ihex)

    input_infos = session.get_input_infos()
    if not inputs:
        rnd_gen = np.random.RandomState(args.seed)
        inputs = gen_datas(runner, session.name, args.batch, args.range, logger, rnd_gen)
    else:
        for idx, info in enumerate(input_infos):
            inputs[idx] = info['io_tensor'].quantize(inputs[idx])

    if args.io_only:
        mode = AiRunner.Mode.IO_ONLY
    elif args.with_data:
        mode = AiRunner.Mode.PER_LAYER_WITH_DATA
    else:
        mode = AiRunner.Mode.PER_LAYER

    if args.perf_only:
        mode |= AiRunner.Mode.PERF_ONLY
    if args.target_log or args.debug:
        mode |= AiRunner.Mode.DEBUG

    if args.range and len(args.range) == 1:
        mode |= AiRunner.Mode.FIXED_INPUT

    logger.info('Invoking STM AI model... (requested mode: {})'.format(mode))

    ai_outputs_s = None
    if args.f_inject:
        os.makedirs("./fault_campaign", exist_ok=True)
        os.makedirs("./fault_campaign/ref", exist_ok=True)

        if h5_model is not None:
            ref = invoke_h5_model(h5_model, inputs)
            ref_shape = ref.shape
            save_tensors(inputs, None, ref, logger, dir_="./fault_campaign/ref")
            ref = ref.argmax(axis=1)
            save_classifications(ref, logger, dir_="./fault_campaign/ref")

            logger.info(f'first5 ref - {ref[:5]}')
        
        if os.path.exists('checkpoint.npy'):
            logger.info('Loading checkpoint data...')
            ai_outputs_s = np.load('checkpoint.npy', allow_pickle=True)
            batch_exec = ai_outputs_s[0].shape[0]
            inputs_s = inputs
            inputs[0] = inputs[0][batch_exec:]
            warm_started = True
        else:
            warm_started = False

        logger.info('Starting fault injection campaign')
        f_campaign_results = {}
        f_bit_positions = gen_f_bit_positions(args.f_bit_range, args.f_bit_start, args.f_bit_step)

        stai_start_time = perf_counter()
        # TODO we can pass only the amount of faults since the faults are decoded outside
        ai_outputs, ai_profiler = session.invoke(inputs, mode=mode, FI_enable=True, f_w_size=args.f_w_size, f_bit_positions=f_bit_positions)
        
        if warm_started:
            for idx, out_ in enumerate(ai_outputs):
                ai_outputs[idx] = np.append(ai_outputs_s[idx], out_, axis=0)
        
        if ai_outputs[0].size != np.prod(ref_shape) and ai_outputs[0].size > 0:
            ai_outputs = np.array(ai_outputs, dtype=object)
            np.save('checkpoint.npy', ai_outputs, allow_pickle=True)

            print('Checkpoint saved, run the script again to continue the fault injection campaign.')
            return 1
        else:
            logger.info('Computing accuracy for the fault injection campaign')
            future_to_f_loc = {}
            with ThreadPoolExecutor() as executor:
                f_idx = 0
                for w_idx in range(args.f_w_size):
                    f_campaign_results[w_idx] = {}
                    for f_bit in f_bit_positions:
                        os.makedirs(f"./fault_campaign/w{w_idx}_b{f_bit}", exist_ok=True)

                        f_campaign_results[w_idx][f_bit] = ai_outputs[f_idx].reshape(ref_shape)
                        save_tensors(inputs_s, None, f_campaign_results[w_idx][f_bit], logger, dir_=f"./fault_campaign/w{w_idx}_b{f_bit}")
                        f_campaign_results[w_idx][f_bit] = f_campaign_results[w_idx][f_bit].argmax(axis=1)
                        save_classifications(f_campaign_results[w_idx][f_bit], logger, dir_=f"./fault_campaign/w{w_idx}_b{f_bit}")
                        future_to_f_loc[executor.submit(calc_acc, ref, f_campaign_results[w_idx][f_bit])] = (w_idx, f_bit)
                        f_idx += 1

            for future in as_completed(future_to_f_loc):
                w_idx, f_bit = future_to_f_loc[future]
                acc = future.result()
                f_campaign_results[w_idx][f_bit] = acc
                logger.info(f'Fault in bit {f_bit} of the weight {w_idx} - Accuracy: {acc}')
        stai_end_time = perf_counter()

        with open("out_dict.txt", 'w') as f:
            f.write(str(f_campaign_results))

    return 0


def main():
    """Main function to parse the arguments"""  # noqa: DAR101,DAR201,DAR401

    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='AI runner')
    parser.add_argument('--desc', '-d', metavar='STR', type=str, help='connection descriptor', default=_DEFAULT)
    parser.add_argument('--list', '-l', action='store_true', help="List the available c-models")
    parser.add_argument('--batch', '-b', metavar='INT', type=int, help='batch_size', default=1)
    parser.add_argument('--seed', metavar='INT', type=int, help='seed value', default=42)
    parser.add_argument('--range', type=float,
                        metavar='RANGE', nargs='+', default=None,
                        help="requested range to generate randomly the input data (default=[0,1])")
    parser.add_argument('--valinput', '-vi', metavar='STR', type=str,
                        help='npy/npz files used as input/output data', default=None)
    parser.add_argument('--image', metavar='FILE', type=str, help='Simple file image is used as input', default=None)
    parser.add_argument('--show', action='store_true', help="Display the loaded image")
    parser.add_argument('--rescale-01', action='store_true',
                        help="Rescale the loaded image between 0 and 1 else -1, 1 is used")
    parser.add_argument('--classifier', action='store_true', help="Indicate that the mode is a classifier")
    parser.add_argument('--io-only', action='store_true', help="The info per layer are not reported")
    parser.add_argument('--perf-only', action='store_true', help="Only the perf info, no data are exchanged")
    parser.add_argument('--with-data', action='store_true', help="dump the intermediate results")
    parser.add_argument('--name', '-n', metavar='STR', type=str, help='c-model name', default=None)
    parser.add_argument('--h5-file', metavar='STR', type=str, help='h5 model file', default=None)
    parser.add_argument('--histo', action='store_true', help="Compute the histogram of the errors (integer data type)")
    parser.add_argument('--histo-all', action='store_true', help="Compute the histogram of the associated REF")
    parser.add_argument('--map', metavar='STR', nargs='+', type=str, help='Output Tensor mapping', default=None)
    parser.add_argument('--ihex', '-x', type=str, nargs='+', help="Intel HEX files")
    parser.add_argument('--verbosity', '-v',
                        nargs='?', const=1,
                        type=int, choices=range(0, 3),
                        help="set verbosity level",
                        default=0)
    parser.add_argument('--debug', action='store_true', help="debug option")
    parser.add_argument('--dequant', action='store_true', help="dequant the values before to compute the metrics")
    parser.add_argument('--no-internals', action='store_true',
                        help="disable the experimental feature to dump the internal tflite tensors")
    parser.add_argument('--tflite-op', type=str, choices=['default', 'auto', 'builtin_ref', 'builtin'],
                        default='auto',
                        help="tflite experimental - select the OpResolverType (default=\'default\')")
    parser.add_argument('--tflite-no-multi-threads', action='store_true',
                        help="Disable the multi-thread support to execute the TFLite model")
    parser.add_argument('--target-log', action='store_true', help="enable additional log from the target")

    parser.add_argument('--f-bit-range', type=int, default=16, help="Fault injection bit range")
    parser.add_argument('--f-bit-start', type=int, default=63, help="Fault injection start bit")
    parser.add_argument('--f-bit-step', type=int, default=32, help="Fault injection bit step")
    parser.add_argument('--f-w-size', type=int, default=730, help="Fault injection weigth size")
    parser.add_argument('--f-inject', action='store_true', help="Enable fault injection")
    args = parser.parse_args()
    logging.basicConfig(format='%(message)s', stream=sys.stdout,
                        level=os.environ.get("LOGLEVEL", "DEBUG" if args.debug else "INFO"))

    if args.histo_all or args.map:
        args.with_data = True

    if not args.with_data and not args.no_internals:
        args.no_internals = True

    return run(args, log)


if __name__ == '__main__':
    sys.exit(main())