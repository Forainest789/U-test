"""Counterfactual utility measurement on a frozen, published memory system.

This package does not train anything and does not modify SlotMem. It intervenes on the
memory condition of the surrounding fork and scores what comes out, because the question
-- when does reading a memory help or hurt the generated video -- only has meaning on a
memory channel that already works. Measuring it on our own untrained injector is what
produced a month of results that could not separate "memory is useless" from "our
injection is inert".
"""
