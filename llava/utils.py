import datetime
import logging
import logging.handlers
import os
import sys

import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None
import torch
import torch, json, random, textwrap, html
from pathlib import Path

class LLMLogger:
    """Lightweight probe that prints one sample every `every` steps."""
    def __init__(self, tokenizer, every=100, out_file="llm_debug.log"):
        self.tok = tokenizer
        self.every = every
        self.file = Path(out_file).open("a")

    def _nice(self, ids):
        # human‑readable decode, keeps special tokens
        return self.tok.decode(ids, skip_special_tokens=False)\
                       .replace("\n", "\\n")[:300]

    def log(self, step, input_ids, inputs_embeds,
                   labels, attention_mask, position_ids):
        if step % self.every:                      # log sparsely
            return
        bs = labels.size(0)
        i  = random.randrange(bs)                  # pick a random sample
        tok_in  = input_ids[i].tolist() if input_ids is not None else None
        txt_in  = self._nice(tok_in) if tok_in else "<embed‑only>"
        txt_lbl = self._nice([ t if t != -100 else self.tok.pad_token_id
                               for t in labels[i].tolist() ])

        # basic sanity flags
        pad_ok  = torch.all(labels[i][attention_mask[i]==0] == -100).item()
        img_ok  = torch.all(labels[i][input_ids[i]==self.tok.convert_tokens_to_ids("<image>")] == -100).item() \
                  if input_ids is not None else True
        pos_mon = (position_ids is None or
                   torch.all(position_ids[i].diff()[attention_mask[i,1:]==1] >= 0).item())

        record = dict(
            step = int(step),
            seq_len = int(attention_mask[i].sum()),
            n_img = int((input_ids[i] == self.tok.convert_tokens_to_ids("<image>")).sum())
                    if input_ids is not None else 0,
            pad_mask_ok   = bool(pad_ok),
            image_mask_ok = bool(img_ok),
            pos_monotone  = bool(pos_mon),
            input  = txt_in,
            target = txt_lbl
        )
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()
        print("[LLM‑LOG]", textwrap.shorten(record["input"], width=120))

llm_logger = None        # will hold the singleton

def dump_stats(t, name):
    if isinstance(t, torch.Tensor):
        print(f"[DEBUG] {name:<20} "
              f"shape={list(t.shape)}, "
              f"μ={t.mean():+.2e}, σ={t.std():.2e}, "
              f"min={t.min():+.2e}, max={t.max():+.2e}")

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
