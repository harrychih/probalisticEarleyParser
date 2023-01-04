#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
from ctypes import Structure
import logging
import math
from string import printable
import struct
from xmlrpc.client import Boolean
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple, Union
from copy import copy


log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
log.disabled = True

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    args = parser.parse_args()
    return args


class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self.rootItem: Item
        self.rootItemList: List[Item] = []
        self.minWeight = float('inf')

        self._run_earley()    # run Earley's algorithm to construct self.cols
    
    def minWeightItem(self) -> None:
        '''
        Get the item that carries the minimum weight, which is the best option
        '''
        # print(rootItems)
        # self.rootItemList.sort(key=lambda x: x.weight)
        self.rootItem = min(self.rootItemList, key=lambda x: x.weight)
        self.minWeight = self.rootItem.weight
       
    
    def output(self) -> tuple:   
        # get the root item with minimum weight
        # use the root item to backtrach and print out the desire output
        # print(lessWeightItem)
        self.minWeightItem()
        root = self.rootItem
        if self.minWeight == float('inf'):
            return None
        res = self.print_item(root)
        return (res, self.minWeight)
    
    def print_item(self, item: Item) -> str:
        q = deque()
        def recur_print(item: Union[Item, List[Item]]):
            if type(item) == list:
                if item[0] is None:
                    return
                for it in item:
                    # print(f"get into {it}")
                    recur_print(it)
            else:
                if item.rule.lhs == self.grammar.start_symbol:
                    # print(f"reach to the root: {item}")
                    q.append(f'({item.rule.lhs} ')
                    recur_print(item.backPointer)
                    q.append(")")
                elif len(item.rule.rhs) == 1 and self.grammar.is_terminal(item.rule.rhs[0]):
                    q.append(f'({item.rule.lhs} {item.rule.rhs[0]})')
                    #recur_print(item.backPointer)
                    # log.debug(f'added the terminal ({item.rule.lhs}({item.rule.rhs[0]}) \t {item}')
                elif len(item.rule.rhs) == 0:
                    # Terminal in the middle of the rule
                    q.append(f' {item.rule.lhs} ')
                    # log.debug(f'added the terminal in the middle of the rule: {item.rule.lhs}')
                    #recur_print(item.backPointer)                  
                else:
                    # print(f"reach to a nonterminal {item}")
                    # Nonterminal in the rule
                    q.append(f'({item.rule.lhs} ')
                    # log.debug(f'added nonterminal ({item.rule.lhs}')
                    recur_print(item.backPointer)
                    q.append(")")

                        
        recur_print(item)
        res = ""
        while q:
            res += q.popleft()

        return res

    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        # initialize a list to store all rootItems so that we can backtrach afterward
        for item in self.cols[-1].all():    # the last column
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0
                    self.rootItemList.append(item)
        if self.rootItemList:
            return True
        return False


       

    def _run_earley(self) -> None:
        """Fill in the Earley chart"""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]
 
        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            while column:    # while agenda isn't empty
                item = column.pop()   # dequeue the next unprocessed item
                next = item.next_symbol();
                if item.flag:
                    if next is None:
                        # Attach this complete constituent to its customers
                        # log.debug(f"{item} => ATTACH")
                        self._attach(item, i)   
                    elif self.grammar.is_nonterminal(next):
                        # Predict the nonterminal after the dot
                        # log.debug(f"{item} => PREDICT")
                        self._predict(next, i)
                    else:
                        # Try to scan the terminal after the dot
                        # log.debug(f"{item} => SCAN")
                        self._scan(item, i)


    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        for rule in self.grammar.expansions(nonterminal):
            bpLen = len(rule.rhs)
            bp = [None for _ in range(bpLen)]
            new_item = Item(rule, dot_position=0, start_position=position, backPointer=bp, weight=rule.weight)
            self.cols[position].push(new_item)
            # log.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            # nonterminal
            # log.debug(f"item is {item}")
            if self.grammar.is_terminal(item.next_symbol()) and len(item.rule.rhs) != 1: 
                new_item = item.with_dot_advanced_terminal(position)
            # terminal in the mid of a rule's rhs
            else:
                new_item = item.with_dot_advanced(None, False)
            # For Debugging
            self.cols[position + 1].push(new_item)
            # log.debug(f"\tScanned to get: {new_item} in column {position+1}")
            # log.debug(f"\t{new_item.backPointer} is now added to the backpointer of {new_item}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        
        mid = item.start_position   # start position of this item = end position of item to its left
        # for customer in self.cols[mid].all():  # could you eliminate this inefficient linear search?
        # selfIdx = -1
        # if item in self.cols[mid]._index:
        #     selfIdx = self.cols[mid]._index[item]
        if item.rule.lhs in self.cols[mid]._next_symbol_index:
            # if customer.next_symbol() == item.rule.lhs:
            for idx in self.cols[mid]._next_symbol_index[item.rule.lhs]:
                # if idx != selfIdx:
                customer = self.cols[mid].take(idx)
                # log.debug(f"customer is {customer}, item is {item}")
                # log.debug(self.cols[mid])
                new_item = customer.with_dot_advanced(item, True)
                self.cols[position].push(new_item)
                # log.debug(f"\tAttached to get: {new_item} in column {position}")
                # log.debug(f"\t{new_item.getBackPointer()} is now added to the backpointer of {new_item}")
                self.profile["ATTACH"] += 1



        
class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.  

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.

    # >>> a = Agenda()
    # >>> a.push(3)
    # >>> a.push(5)
    # >>> a.push(3)   # duplicate ignored
    # >>> a
    # Agenda([]; [3, 5])
    # >>> a.pop()
    # 3
    # >>> a
    # Agenda([3]; [5])
    # >>> a.push(3)   # duplicate ignored
    # >>> a.push(7)
    # >>> a
    # Agenda([3]; [5, 7])
    # >>> while a:    # that is, while len(a) != 0
    # ...    print(a.pop())
    # 5
    # 7

    """

    def __init__(self) -> None:
        self._items: List[Item] = []       # list of all items that were *ever* pushed
        self._next = 0                     # index of first item that has not yet been popped
        self._index: Dict[Item, int] = {}  # stores index of an item if it has been pushed before
        self._next_symbol_index: Dict[Item.next_symbol(), set(int)] = defaultdict(set)
        self._lhs_symbol_index: Dict[Item.rule.lhs, set(int)] = defaultdict(set)
        # # self._rhs_tuple_index: Dict[Tuple(Item.dot_position, Item.rule.rhs), set(int)] = defaultdict
        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.  
        # 
        # However, we provided this design because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    def push(self, item: Item) -> None:
        """Add (enqueue) the item, unless it was previously added."""
        if item not in self._index:    # O(1) lookup in hash table
            # Check Duplicates only when push a predicted item or a item, whose right hand side has been fullly scanned
            self._items.append(item)
            idx = len(self._items) - 1
            self._index[item] = idx
            self._next_symbol_index[item.next_symbol()].add(idx)
            self._lhs_symbol_index[item.rule.lhs].add(idx)
        else:
            addedItemIdx = self._index[item]
            addedItem = self.take(addedItemIdx)
            if item.weight < addedItem.weight:
                addedItem.flag = False
                self._next_symbol_index[addedItem.next_symbol()].remove(addedItemIdx)
                self._lhs_symbol_index[addedItem.rule.lhs].remove(addedItemIdx)
                self._items.append(item)
                idx = len(self._items) - 1
                self._index[item] = idx
                self._next_symbol_index[item.next_symbol()].add(idx)
                self._lhs_symbol_index[item.rule.lhs].add(idx)
  

    def take(self, index: int) -> Item:
        return self._items[index]
            
    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self)==0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item
    
    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return [item for item in self._items if item.flag]

    def __repr__(self):
        """Provide a REPResentation of the instance for printing."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"

class Grammar:
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it

        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited linfore of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())  
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions

    def is_terminal(self, symbol: str) -> bool:
        """Is symbol a terminal symbol?"""
        return symbol not in self._expansions
    


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us specify that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    Convenient abstraction for a grammar rule. 
    A rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0
    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        return f"{self.lhs} → {' '.join(self.rhs)}"

    
# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=False)
class Item:
    """An item in the Earley parse table, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    # backPointer to track the previous item
    backPointer: Union[List[Item], List[None]]
    # weight for this item
    weight: float
    flag: Boolean = True
    
    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced_terminal(self, position: int) -> Item:
        rule = Rule(lhs=self.rule.rhs[self.dot_position], rhs=tuple(), weight=0)
        terminalItem = Item(rule, dot_position=0, start_position=position, backPointer=[None], weight=rule.weight, flag=True)
        bp = copy(self.backPointer)
        bp[self.dot_position] = terminalItem
        new_item = Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position, backPointer=bp, weight=self.weight, flag=True)
        return new_item

    def with_dot_advanced(self, item: Union[Item, None], add_weight: Boolean) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        assert (item is None) == (not add_weight)
        if item is None and not add_weight:
            weight = self.weight
        else:
            weight = self.weight + item.weight
        if add_weight:
            # Attach
            # e.g.
            # Self: S -> NP . VP item: VP -> V NP .
            # Self: S -> . NP  VP  item: NP -> PaPa .
            # log.debug(f"self is {self}")
            # log.debug(f"item is {item}")
            bp = copy(self.backPointer)
            bp[self.dot_position] = item
            # log.debug(f"attach the item to the old backpointers, we have {bp}")
            new_item =  Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position, backPointer=bp, weight=weight, flag=True)
            return new_item    
        else:
            # Scan
            # e.g. 
            # self: N -> . Papa
            # new_item: N -> Papa .
            new_item = Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position, backPointer=[None], weight=weight, flag=True)
            
            # log.debug(f'new item {new_item} is added to the backPointers of {self}')
            return new_item
        

    def __repr__(self) -> str:
        """Complete string used to show this item at the command line"""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule}, {self.weight}, {self.backPointer})"  # matches notation on slides

    def __eq__(self, __o: Item) -> bool:
        if (self.rule == __o.rule and self.dot_position == __o.dot_position and self.start_position == __o.start_position):
            return True
    
    def __hash__(self) -> int:
        return hash((self.rule, self.start_position, self.dot_position))


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.verbose)  # Set logging level appropriately

    grammar = Grammar(args.start_symbol, args.grammar)
    
    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug("="*70)
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                if chart.accepted(): 
                    res, weight = chart.output()
                    print(res)
                    print(weight)
                else:
                    print('NONE')
                # )
                log.debug(f"Profile of work done: {chart.profile}")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)   # run tests
    main()
