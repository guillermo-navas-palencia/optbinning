Hi Guillermo — I've opened a batch of PRs against several of the open issues (mine and a few related ones I ran into along the way): #389, #390, #391, #392, #393, #394, #395, #396, #397, #398, #401, #402, #405, #406, #407, #408, #409.

Since several of them touch the same few files, I test-merged all of them against each other locally (actual `git merge` trials for every pair, not just eyeballing diffs) to see where they'd actually conflict, in case it's useful for deciding review/merge order. Happy to rebase/adjust on my end wherever it helps rather than leaving it to GitHub's conflict UI.

Two things I found and already fixed on my end, no action needed from you:
- #390 was originally branched off #389, so its diff was temporarily showing #389's fix duplicated. Rebased it onto master directly and moved where its new tests get inserted — merges cleanly against #389 now, either order.
- #395 and #408 both added code right after the same line in `Scorecard._fit()` (plus a smaller import-line overlap with #398). Moved where #408 inserts its block/import/test — no code changed, just relocated — so both now merge clean against #395 and #398 in either order.

What's left, from the full pairwise scan (every one of the 136 possible pairs tested): 12 conflicting pairs, all of them the same trivial pattern — two PRs independently appending a new test (or import) at the same anchor line in a shared test file. None of them touch source code or change behavior either way; each is a few seconds with GitHub's "keep both changes" button. They cluster in two spots:

1. `tests/test_binning.py` / `test_continuous_binning.py` / `test_multiclass_binning.py` — #389, #390, #392, #394, #407 all append tests near the end of these files, so most pairs among those five will show a conflict when merged one after another.
2. `tests/test_binning_process.py` / `test_scorecard.py` — #396, #398, #401, #402 have the same pattern, more scattered (#398 conflicts with each of the other three; #401×#402 also conflicts).

Suggested order, front-loading what's conflict-free:

1. **#391, #393, #395, #397, #405, #406, #408, #409** — clean against everything else, any order.
2. **#389, #390, #392, #394, #407** — work through these one at a time; each new one will likely flag a trivial "both added a test here" conflict against one or two of the earlier ones in this group — keep both.
3. **#396, #398, #401, #402** — same story, #398 is the one that overlaps with each of the other three.

If you'd rather I clean these up preemptively instead of you hitting them in the UI, say the word and I'll go relocate the remaining test insertions the same way I did for #390/#395/#408 above — it's a well-worn pattern at this point. Otherwise I'll leave them as-is since they're genuinely trivial either way.

One correctness issue worth flagging separately (not a git conflict, so it won't show up anywhere in GitHub's UI): #409 renames the internal resolved-dtype attribute from `self.dtype` to `self._dtype`. #389's `read_json()` restores the other post-fit attributes needed for `transform()` to work after a reload, but was naturally written before `self._dtype` existed, so once both are merged, `transform()` on a `read_json()`-reloaded object will silently return wrong values instead of erroring (`self._dtype` stays `None`). I can't fix this as part of either PR alone since it only exists once both are on master — happy to submit a two-line follow-up (`self._dtype = bin_table_attr['dtype']` in `read_json()`, `binning.py` and `continuous_binning.py`) right after both land, unless you'd rather fold it into #389 or #409 directly during review.

No pressure on the order above — just sharing what I found so you're not rediscovering it PR by PR. Let me know if you'd like me to combine or split any of these differently, or if it'd be easier for you if I merged some of the smaller/related ones into a single PR.
