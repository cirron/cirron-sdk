"""
Parser for selector expressions in string format.

This module provides functionality to parse string-based selector expressions
like "numeric() & ~regex('^temp_')" into actual Selector objects.
"""

import re
from typing import Union, Dict, List, Optional, Any
from .selectors import (
    Selector, ColumnSelector, TypeSelector, RegexSelector, TagSelector,
    FunctionSelector, UnionSelector, IntersectionSelector, NotSelector,
    numeric, categorical, datetime, text, boolean, integer, float_type,
    regex, tags, columns, all_columns, none, custom
)
import logging

logger = logging.getLogger(__name__)


class SelectorParseError(Exception):
    """Exception raised when selector expression parsing fails."""
    pass


class SelectorParser:
    """Parser for string-based selector expressions.
    
    Supports expressions like:
    - "numeric()"
    - "categorical() & ~regex('^temp_')"
    - "tags('geo', 'location') | columns('lat', 'lng')"
    - "(numeric() | datetime()) & ~columns('id')"
    """
    
    def __init__(self, tag_mapping: Optional[Dict[str, List[str]]] = None):
        """Initialize parser with optional tag mapping.
        
        Args:
            tag_mapping: Optional mapping from tag names to column lists
        """
        self.tag_mapping = tag_mapping or {}
        
        # Define function mappings
        self.functions = {
            'numeric': numeric,
            'categorical': categorical,
            'datetime': datetime,
            'text': text,
            'boolean': boolean,
            'integer': integer,
            'float': float_type,
            'regex': self._parse_regex,
            'tags': self._parse_tags,
            'columns': self._parse_columns,
            'all': all_columns,
            'none': none,
        }
    
    def parse(self, expression: str) -> Selector:
        """Parse a selector expression string into a Selector object.
        
        Args:
            expression: String expression to parse
            
        Returns:
            Parsed Selector object
            
        Raises:
            SelectorParseError: If parsing fails
        """
        try:
            # Clean up the expression
            expression = expression.strip()
            if not expression:
                return all_columns()
            
            # Parse the expression
            return self._parse_expression(expression)
            
        except Exception as e:
            raise SelectorParseError(f"Failed to parse selector expression '{expression}': {e}")
    
    def _parse_expression(self, expr: str) -> Selector:
        """Parse a full expression with operators."""
        return self._parse_or_expression(expr)
    
    def _parse_or_expression(self, expr: str) -> Selector:
        """Parse OR expressions (|)."""
        parts = self._split_by_operator(expr, '|')
        if len(parts) == 1:
            return self._parse_and_expression(parts[0])
        
        selectors = [self._parse_and_expression(part) for part in parts]
        return UnionSelector(*selectors)
    
    def _parse_and_expression(self, expr: str) -> Selector:
        """Parse AND expressions (&)."""
        parts = self._split_by_operator(expr, '&')
        if len(parts) == 1:
            return self._parse_not_expression(parts[0])
        
        selectors = [self._parse_not_expression(part) for part in parts]
        return IntersectionSelector(*selectors)
    
    def _parse_not_expression(self, expr: str) -> Selector:
        """Parse NOT expressions (~)."""
        expr = expr.strip()
        if expr.startswith('~'):
            inner_expr = expr[1:].strip()
            return NotSelector(self._parse_atom(inner_expr))
        else:
            return self._parse_atom(expr)
    
    def _parse_atom(self, expr: str) -> Selector:
        """Parse atomic expressions (functions, parentheses)."""
        expr = expr.strip()
        
        # Handle parentheses
        if expr.startswith('(') and expr.endswith(')'):
            return self._parse_expression(expr[1:-1])
        
        # Handle function calls
        func_match = re.match(r'(\w+)\s*\((.*)\)', expr)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            return self._parse_function_call(func_name, args_str)
        
        # Handle simple function names without parentheses
        if expr in self.functions:
            return self.functions[expr]()
        
        raise SelectorParseError(f"Unknown expression: '{expr}'")
    
    def _parse_function_call(self, func_name: str, args_str: str) -> Selector:
        """Parse a function call with arguments."""
        if func_name not in self.functions:
            raise SelectorParseError(f"Unknown function: '{func_name}'")
        
        func = self.functions[func_name]
        
        # Parse arguments
        if not args_str.strip():
            return func()
        
        # For now, use a simple argument parser
        # In production, you might want a more sophisticated parser
        args = self._parse_arguments(args_str)
        
        try:
            return func(*args)
        except Exception as e:
            raise SelectorParseError(f"Error calling {func_name}({args_str}): {e}")
    
    def _parse_arguments(self, args_str: str) -> List[str]:
        """Parse function arguments from string."""
        args = []
        current_arg = ""
        in_quotes = False
        quote_char = None
        paren_depth = 0
        
        for char in args_str:
            if not in_quotes:
                if char in ['"', "'"]:
                    in_quotes = True
                    quote_char = char
                    current_arg += char
                elif char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    args.append(self._clean_argument(current_arg))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == quote_char:
                    in_quotes = False
                    quote_char = None
        
        if current_arg:
            args.append(self._clean_argument(current_arg))
        
        return args
    
    def _clean_argument(self, arg: str) -> str:
        """Clean and process a single argument."""
        arg = arg.strip()
        
        # Remove quotes if present
        if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
            arg = arg[1:-1]
        
        return arg
    
    def _split_by_operator(self, expr: str, operator: str) -> List[str]:
        """Split expression by operator, respecting parentheses."""
        parts = []
        current_part = ""
        paren_depth = 0
        in_quotes = False
        quote_char = None
        i = 0
        
        while i < len(expr):
            char = expr[i]
            
            if not in_quotes:
                if char in ['"', "'"]:
                    in_quotes = True
                    quote_char = char
                elif char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == operator and paren_depth == 0:
                    parts.append(current_part)
                    current_part = ""
                    i += 1
                    continue
            else:
                if char == quote_char:
                    in_quotes = False
                    quote_char = None
            
            current_part += char
            i += 1
        
        if current_part:
            parts.append(current_part)
        
        return [part.strip() for part in parts if part.strip()]
    
    def _parse_regex(self, pattern: str, ignore_case: str = "True") -> RegexSelector:
        """Parse regex function call."""
        ignore_case_bool = ignore_case.lower() in ['true', '1', 'yes', 'on']
        return regex(pattern, ignore_case_bool)
    
    def _parse_tags(self, *tag_names: str) -> TagSelector:
        """Parse tags function call."""
        return tags(*tag_names, tag_mapping=self.tag_mapping)
    
    def _parse_columns(self, *column_names: str) -> ColumnSelector:
        """Parse columns function call."""
        return columns(*column_names)


def parse_selector(expression: Union[str, Selector, List[str]], 
                  tag_mapping: Optional[Dict[str, List[str]]] = None) -> Selector:
    """Parse a selector from various input formats.
    
    Args:
        expression: Selector expression (string, Selector object, or list of column names)
        tag_mapping: Optional tag mapping for TagSelector
        
    Returns:
        Parsed Selector object
    """
    if isinstance(expression, Selector):
        return expression
    elif isinstance(expression, str):
        parser = SelectorParser(tag_mapping)
        return parser.parse(expression)
    elif isinstance(expression, (list, tuple)):
        return ColumnSelector(list(expression))
    else:
        raise SelectorParseError(f"Cannot parse selector from type: {type(expression)}")


def validate_selector_expression(expression: str, 
                                tag_mapping: Optional[Dict[str, List[str]]] = None) -> bool:
    """Validate that a selector expression is parseable.
    
    Args:
        expression: Selector expression string
        tag_mapping: Optional tag mapping
        
    Returns:
        True if expression is valid, False otherwise
    """
    try:
        parse_selector(expression, tag_mapping)
        return True
    except SelectorParseError:
        return False


def get_selector_help() -> str:
    """Get help text for selector expressions."""
    return """
Selector Expression Syntax:

Basic Selectors:
  numeric()           - Select numeric columns
  categorical()       - Select categorical columns  
  datetime()          - Select datetime columns
  text()              - Select text/string columns
  boolean()           - Select boolean columns
  integer()           - Select integer columns
  float()             - Select float columns
  columns('a', 'b')   - Select specific columns by name
  regex('^feat_')     - Select columns matching regex pattern
  tags('geo', 'meta') - Select columns with specified tags
  all()               - Select all columns
  none()              - Select no columns

Operators:
  &                   - AND (intersection)
  |                   - OR (union)
  ~                   - NOT (inversion)
  ()                  - Grouping

Examples:
  "numeric()"                           - All numeric columns
  "numeric() & ~regex('^temp_')"        - Numeric columns not starting with 'temp_'
  "categorical() | datetime()"          - Categorical or datetime columns
  "tags('geo') & numeric()"             - Numeric columns tagged as 'geo'
  "(numeric() | text()) & ~columns('id')" - Numeric or text columns except 'id'
"""