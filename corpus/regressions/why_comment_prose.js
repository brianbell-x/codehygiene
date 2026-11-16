function visitSelectorList(node, state, parent) {
  if (state.pruning) {
    state.code.remove(node.start, node.end);
  }

  // if we are inside an `:is(...)` wrapper, keep the existing specificity bump state
  let specificity = state.specificity;

  // a selector list that belongs to a rule needs a specificity bump for the
  // first scoped selector, but only when the rule sits at the top level
  if (parent?.type === 'Rule') {
    specificity = { bumped: false };
  }
  return specificity;
}
