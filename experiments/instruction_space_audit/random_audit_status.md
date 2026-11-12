# 随机审计进度

规则在 `random_audit_protocol.md`。抽样框是 538 条非候选路线，种子 `20261022`，简单随机 100 条，记录在 `random_audit_sample.csv`。抽中的构成是部分成功 51、三句都成功 31、三句都失败 18，其中 4 条没有 Go2 路径。

序号 1 到 100 已按同一套规则标完。序号 1 到 6 的标签没有改。结果：

- 路线：确认冲突 68，不确定 25，两者都没有 7。标签范围 68% 到 93%。
- 指令：yes 122，uncertain 96，no 52，wording_only 30。标签范围 40.7% 到 72.7%。
- `explains_failure=yes` 为 0。有轨迹的确认冲突只有 episode 393，判定是 `no`。

汇总写在 `random_audit_report.md`。路线级在 `random_audit_route_results.csv`，指令级在 `random_audit_labels.csv`，短语级在 `random_audit_phrases.csv` 和 `pilot_phrases.csv`。

这 100 条只估计 538 条非候选路线。75 条显式转向候选还没有做这套语义审计。

冻结标签之外还有一轮严格复核，见 `strict_conflict_review.md`。122 条被反驳短语中，61 条仍是直接冲突，46 条现有证据不够支持直接冲突，15 条不能定。原来的 92 条“目标在转向后才到前方”里，46 条属于后者。`semantic_conflict_confirmed` 没有改。
