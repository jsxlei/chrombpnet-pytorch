#!/usr/bin/env python

# Author: Lei Xiong <jsxlei@gmail.com>

import logging

def create_logger(name='', ch=True, fh=False, levelname=logging.INFO, overwrite=False):
    logger = logging.getLogger(name)
    logger.setLevel(levelname)
    
    if overwrite:
        for h in logger.handlers:
            logger.removeHandler(h)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler 
    if ch:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if fh:
        fh = logging.FileHandler(fh, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger