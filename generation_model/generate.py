import time
import torch
import fileinput

from fairseq.data import encoders
from collections import namedtuple
from fairseq import checkpoint_utils, options, tasks, utils

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(ipt, buffer_size):
    buffer = []
    with fileinput.input(files=[ipt], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def load_model(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    base_dict = {
        'args': args,
        'task': task,
        'max_positions': max_positions,
        'encode_fn': encode_fn,
        "decode_fn": decode_fn,
        'use_cuda': use_cuda,
        'generator': generator,
        'models': models,
        'tgt_dict': tgt_dict,
        'src_dict': src_dict,
        "align_dict": align_dict
    }

    return base_dict


class Gen:
    # def __init__(self, data='preprocessed-data',
    #              path='checkpoints/zh-cp-wuqing/checkpoint31.pt',
    #              beam=20, nbest=20, cpu=True):
    def __init__(self, data='./generation_model/preprocessed-data',
                 path='./generation_model/checkpoints/zh-cp-wuqing/checkpoint31.pt',
                 beam=20, nbest=20, cpu=True):
        parser = options.get_generation_parser(interactive=True)
        args = options.parse_args_and_arch(parser, input_args=[data])
        args.path = path
        args.beam = beam
        args.nbest = nbest
        args.cpu = cpu
        self.gen_utils = load_model(args)

    def gen(self, ipt):
        start_id = 0
        results = []
        inputs = [ipt]
        for batch in make_batches(inputs, self.gen_utils["args"], self.gen_utils["task"],
                                  self.gen_utils["max_positions"],
                                  self.gen_utils["encode_fn"]):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.gen_utils["use_cuda"]:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.gen_utils["task"].inference_step(self.gen_utils["generator"], self.gen_utils["models"],
                                                                 sample)
            for i, (idx, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.gen_utils["tgt_dict"].pad())
                results.append((start_id + idx, src_tokens_i, hypos))
        outputs = []
        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            src_str = self.gen_utils["src_dict"].string(src_tokens, self.gen_utils["args"].remove_bpe)
            print('S-{}\t{}'.format(id, src_str))
            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.gen_utils["args"].nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=self.gen_utils["align_dict"],
                    tgt_dict=self.gen_utils["tgt_dict"],
                    remove_bpe=self.gen_utils["args"].remove_bpe,
                )
                hypo_str = self.gen_utils["decode_fn"](hypo_str)
                outputs.append(hypo_str)
        # update running id counter
        start_id += len(inputs)
        return outputs


# if __name__ == '__main__':
#     # 先创建对象，加载模型
#     # 这一步必须单独提出来，在项目的启动的时候就加载
#     gen_model = Gen()
#     st = time.time()
#     sec_set = gen_model.gen("听 泉 石 上 花 无 语")
#     print(sec_set)
#     ed = time.time()
#     print("生成消耗时间:", ed - st)   # < 1s
