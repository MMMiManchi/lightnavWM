#!/usr/bin/env python3
"""Write the strict semantic review without changing frozen labels.

The frozen verdict stays in random_audit_phrases.csv. This script only
writes strict_conflict_review.csv and strict_conflict_review_instructions.csv.
Success and failure are not read.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# episode_id, phrase_index, constraint type, strict status, note.
# Types: immediate_action, destination_description, spatial_relation,
# stop_condition, unclear.
# Status: confirmed, not_confirmed, uncertain.
REVIEWS = {
    (1165, 1): ("destination_description", "not_confirmed", "只写走进并穿过厨房，没有规定第一步直行。先转身再进入厨房，和这句话相容。"),
    (1166, 2): ("immediate_action", "confirmed", "写了 go straight。掉头后的下一段是 +60.3°，不是直行进入厨房。"),
    (1167, 1): ("destination_description", "not_confirmed", "只写去厨房再到餐桌，没有规定当前朝向。厨房在转身之后出现，不能单独证明短语为假。"),
    (547, 1): ("immediate_action", "confirmed", "写了 Walk straight。第一段是 -79.6°。"),
    (67, 1): ("spatial_relation", "confirmed", "短语规定高脚椅在右侧。w0 里高脚椅在画面左侧。这是当前画面里的左右关系，不是先转身再找目标。"),
    (68, 1): ("immediate_action", "confirmed", "写了 Walk straight。第一段是 +92.2°。"),
    (68, 2): ("spatial_relation", "confirmed", "短语规定沙发在右侧。终点画面里沙发在左侧。"),
    (400, 2): ("immediate_action", "confirmed", "写了 walk straight。下楼后的下一段是 -48.7°。这是 45° 到 60° 的边界案例，仍按直行被明显打断计。"),
    (402, 2): ("immediate_action", "confirmed", "写了 Walk forward。下楼后的下一段是 -48.7°。"),
    (937, 3): ("stop_condition", "confirmed", "停止条件是门口。终点画面已经在房间内部，面对壁炉和沙发。"),
    (1591, 1): ("immediate_action", "confirmed", "写了 Go straight。第一段是 +121.3°，沙发不在起步正前方。"),
    (1592, 1): ("spatial_relation", "uncertain", "写了从椅子右侧经过。证据只说明起步是 +121.3° 左转，没有在椅子旁边看清路径在哪一侧。"),
    (1593, 1): ("destination_description", "not_confirmed", "只写走到大厅尽头。大厅尽头在两次左转之后，短语没有规定第一步直行。"),
    (1555, 1): ("destination_description", "not_confirmed", "只写经过头顶有吊灯的大餐桌。餐桌在 +172.1° 之后，先转身再经过它，和这句话相容。"),
    (1556, 1): ("destination_description", "not_confirmed", "只写穿过房间、经过桌椅进入走廊，没有写直行。第一段 +172.1° 是去面对餐桌，不是对短语的直接反驳。"),
    (550, 1): ("spatial_relation", "not_confirmed", "只写走最左边的门。起步画面没有门，不能证明转身之后也不存在最左边的那扇门。"),
    (552, 1): ("spatial_relation", "uncertain", "短语用两把椅子的小桌来指定左侧门。起步看到的是长桌，转弯之后有没有这张小桌，现有画面没有核到。"),
    (1456, 1): ("destination_description", "not_confirmed", "只写上楼。楼梯在 +172.7° 之后，随后高度上升。先转身再上楼，和这句话相容。"),
    (1457, 1): ("destination_description", "not_confirmed", "只写上楼。楼梯不在起步画面里，不构成对上楼这句话的直接反驳。"),
    (1457, 2): ("immediate_action", "confirmed", "写了 continue straight。到楼面后的两段是 +35.0° 和 +26.1°，同向累计约 61°，不是直行。"),
    (1458, 1): ("destination_description", "not_confirmed", "只写上到下一层。上楼方向在转身之后，短语没有规定楼梯已经在眼前。"),
    (1771, 2): ("immediate_action", "confirmed", "写了 straight ahead。下一动作是 -80.6°，不是直走进另一间卧室。"),
    (1772, 1): ("destination_description", "not_confirmed", "只写经过床尾并从房间另一侧出门。床在 -173.0° 之后，先转身再经过床尾，和这句话相容。"),
    (1773, 2): ("immediate_action", "confirmed", "写了 Walk forward。离开后的下一段是 -80.6°。"),
    (535, 1): ("destination_description", "not_confirmed", "只写下楼。楼梯在 -169.9° 之后，随后高度下降。先转身再下楼，和这句话相容。"),
    (536, 1): ("destination_description", "not_confirmed", "只写从阁楼下楼。阁楼的床在眼前、楼梯在身后，不表示下楼这句话为假。"),
    (537, 1): ("destination_description", "not_confirmed", "只写下楼。下楼方向不在起步画面里，短语没有规定必须正面朝楼梯出发。"),
    (768, 4): ("immediate_action", "confirmed", "写了 make a right。左转之后的下一段是 +15.7°，路径上没有这次右转。"),
    (1654, 1): ("destination_description", "uncertain", "写走下这条走廊。起步前方能看到玻璃门里的走廊，参考路径第一段却是 -73.9°。短语没有写 straight，现有证据分不清是走廊本身转弯，还是离开了眼前这条走廊。"),
    (1655, 1): ("immediate_action", "confirmed", "写了 to straight。离开后的第一段是 -73.9°。"),
    (358, 5): ("immediate_action", "confirmed", "写了 turn right。直行段之后的下一次转弯是 +65.0° 左转。"),
    (685, 1): ("immediate_action", "confirmed", "写了 Turn right。第一段是 +149.8° 左转，不是在楼梯处右转。"),
    (686, 2): ("immediate_action", "confirmed", "写了 walk up。路径各段没有上升，最后一段高度变化是 -0.69 米。"),
    (687, 1): ("spatial_relation", "confirmed", "短语规定楼梯在右侧。起步画面里栏杆在左侧。"),
    (687, 2): ("immediate_action", "confirmed", "写了 go forward。第一段是 +149.8°。"),
    (42, 3): ("immediate_action", "confirmed", "写了 Turn right。离开厨房通道时的航向变化是 +47.7° 左转。"),
    (926, 5): ("stop_condition", "confirmed", "停止条件是门口。终点画面在卧室内部，看着床和床头柜。"),
    (19, 4): ("immediate_action", "confirmed", "写了 Continue ahead。从玻璃门到拱门的航向变化是 +84.3°。"),
    (415, 1): ("destination_description", "not_confirmed", "只写沿走廊经过镜子。走廊在 +176.8° 之后，短语没有规定从入户门朝向直行。"),
    (417, 1): ("immediate_action", "confirmed", "写了 Walk forward。第一段是 +176.8°，起步对着门，不是走廊。"),
    (145, 1): ("immediate_action", "confirmed", "写了 take a right。第一段是 +58.8° 左转。"),
    (1141, 1): ("spatial_relation", "uncertain", "写床右侧的门。床在 +148.8° 之后才出现，右侧关系没有在床旁边核到。仅凭床不在眼前，不能确定这句话为假。"),
    (1143, 1): ("destination_description", "not_confirmed", "只写进入卧室。卧室在 +148.8° 之后，先转身再进入，和这句话相容。"),
    (28, 1): ("spatial_relation", "not_confirmed", "写从右侧的门离开。第一段是 -149.8°，方向仍是向右。门不在起步画面里，不能单独证明它不在右侧。"),
    (30, 1): ("spatial_relation", "not_confirmed", "写穿过右侧门洞。第一段同样是 -149.8° 向右。现有证据没有显示这个门在左侧。"),
    (1036, 1): ("immediate_action", "confirmed", "写了 Go straight。沙发在 +94.7° 之后，第一段不是直行。"),
    (1441, 1): ("destination_description", "not_confirmed", "写绕过床再进入右侧房间。床在转身之后，后续还有 -58.9° 的右转。目标在身后，不否定绕床向右进门。"),
    (910, 2): ("destination_description", "not_confirmed", "只写沿走廊往前直到左侧另一条走廊。走廊在 +176.8° 之后，短语没有写 straight 或 forward。"),
    (911, 1): ("destination_description", "not_confirmed", "只写穿过走廊。起步对着门，先转身进入走廊，和这句话相容。"),
    (912, 1): ("immediate_action", "confirmed", "写了 Walk forward。第一段是 +176.8°。"),
    (823, 1): ("destination_description", "not_confirmed", "只写进入有白色头像的房间。头像在 +118.4° 之后，短语没有规定它已经在眼前。"),
    (825, 2): ("immediate_action", "confirmed", "短语规定地毯大约向右转 30°。这之后的航向变化只有约 +1° 到 -4°，没有这段右转。"),
    (825, 3): ("stop_condition", "confirmed", "停止条件是地毯完成那段右转之后。路径上没有这段右转，停止条件无法成立。"),
    (1516, 1): ("immediate_action", "confirmed", "写了 Move forward，并规定从右侧脚凳和左侧篮子之间走过。第一段是 -122.0°，不是向前。"),
    (1518, 1): ("immediate_action", "confirmed", "写了 Go straight。第一段是 -122.0°，楼梯不在起步正前方。"),
    (105, 1): ("spatial_relation", "confirmed", "写穿过右侧拱门。到达拱门的转弯是 +172.3° 左转，不是向右。"),
    (610, 1): ("destination_description", "not_confirmed", "只写走到房间远端并穿过双开门。双开门在 +178.6° 之后，先转身再走过去，和这句话相容。"),
    (611, 3): ("stop_condition", "confirmed", "停止条件是刚进门。终点在进门以后约 8 米的房间远端。"),
    (612, 1): ("spatial_relation", "uncertain", "写路径右侧有信息牌。信息牌不在起步画面，右侧关系没有被核到。先转身不能单独证明短语为假。"),
    (541, 1): ("destination_description", "not_confirmed", "只写上完楼梯。楼梯在 +162.4° 之后，随后高度上升。"),
    (542, 1): ("destination_description", "not_confirmed", "只写上楼。上楼方向不在起步画面里，短语没有禁止先转身。"),
    (543, 1): ("destination_description", "not_confirmed", "只写上楼。楼梯在 +162.4° 之后。"),
    (76, 1): ("destination_description", "not_confirmed", "只写穿过门洞进入餐厅。餐厅在 +124.7° 之后，先转身再进入，和这句话相容。"),
    (77, 1): ("destination_description", "uncertain", "写最近的一扇开着的双开门。带柜子的餐厅在 +124.7° 之后。眼前有没有更近的门，现有画面没有清点。"),
    (78, 1): ("destination_description", "not_confirmed", "只写穿过餐厅、经过桌子。餐桌在 +124.7° 之后，不表示这句话要求桌子已经在眼前。"),
    (394, 1): ("immediate_action", "confirmed", "写了 Move to the right。第一段是 +144.2° 左转。"),
    (395, 1): ("destination_description", "not_confirmed", "只写经过起居区走到壁炉。壁炉在 +144.2° 之后，短语没有规定直行。"),
    (396, 1): ("destination_description", "not_confirmed", "只写走到壁炉前面。壁炉不在起步前进方向上，先转身再走到它前面，和这句话相容。"),
    (607, 1): ("spatial_relation", "uncertain", "写从炉灶左侧的门离开。炉灶没有入画，-63.4° 不能证明那扇门不在炉灶左侧。"),
    (794, 5): ("stop_condition", "confirmed", "停止条件是离开走廊之后。终点画面仍在走廊里。"),
    (994, 1): ("immediate_action", "confirmed", "写了 Walk straight。开头两段是 -35.5° 和 -55.5°，同向累计约 91°。"),
    (995, 5): ("immediate_action", "confirmed", "写了 turn right。走廊之后的转弯是 +59.0° 左转。"),
    (899, 4): ("spatial_relation", "confirmed", "写进入右侧第二扇门。进入卧室的转弯是 +70.9° 左转。"),
    (900, 6): ("spatial_relation", "confirmed", "写进入右侧的门。进门的航向变化是 +70.9° 左转。"),
    (900, 7): ("stop_condition", "confirmed", "停止条件是房间门口。终点画面已经在卧室里看着床。"),
    (511, 1): ("destination_description", "not_confirmed", "写离开厨房并朝向小餐厅。小餐厅在 +136.9° 之后。heading towards 允许先转身再朝它走。"),
    (512, 1): ("destination_description", "not_confirmed", "只写离开厨房。离开时第一段是 +136.9°，转身出门仍然是离开厨房。"),
    (513, 1): ("destination_description", "not_confirmed", "只写进入有壁炉的起居区。这个区域在两次转弯之后，短语没有规定它在眼前。"),
    (1424, 4): ("immediate_action", "uncertain", "写了 Continue straight。中间两段是 +17.7° 和 +28.6°，同向累计约 46°。有直行这个词，但角度接近走廊里的弯曲，不足以确认直接冲突。"),
    (556, 1): ("destination_description", "not_confirmed", "只写走楼梯。楼梯在 -169.9° 之后，随后高度下降。先转身再上楼梯或下楼梯，短语本身没有禁止。"),
    (557, 2): ("spatial_relation", "not_confirmed", "写下楼并且楼梯在左侧。下楼时的转弯是 +41.6° 向左。楼梯不在起步画面左侧，不等于下楼时不在左侧。"),
    (558, 1): ("destination_description", "not_confirmed", "只写下楼。下楼方向在 -169.9° 之后。"),
    (1317, 1): ("immediate_action", "confirmed", "写了 Turn around。第一段只有 -60.2°，不是掉头。"),
    (391, 1): ("destination_description", "not_confirmed", "只写走进餐厅。餐厅在 +172.1° 之后，先转身再进入，和这句话相容。"),
    (393, 1): ("destination_description", "not_confirmed", "只写走向红色大桌和椅子，没有规定第一步直行。红桌在 +172.1° 之后。先转身再朝它走，是这句话的一种正常执行。"),
    (112, 1): ("destination_description", "not_confirmed", "只写走到露台另一侧。另一侧在 +144.8° 之后，短语没有规定直行。"),
    (113, 1): ("immediate_action", "confirmed", "写了 walk forward，并规定窗户在右侧。第一段是 +144.8°，窗户和露台沙发不在这个前进方向上。"),
    (114, 1): ("destination_description", "not_confirmed", "只写经过室外起居区。它在 +144.8° 之后，短语没有规定当前朝向。"),
    (202, 1): ("immediate_action", "uncertain", "左转被定位在床尾。起步在浴室，床在 +152.4° 之后。现有画面没有核到床尾处的转弯是不是向左。"),
    (578, 1): ("destination_description", "not_confirmed", "只写沿走廊走向两个白色花瓶。花瓶在 -96.6° 之后，先转身再走向它们，和这句话相容。"),
    (579, 1): ("destination_description", "not_confirmed", "只写经过两个陶罐。陶罐在 -96.6° 之后，短语没有规定它们已经在眼前。"),
    (1030, 4): ("immediate_action", "confirmed", "写了 straight ahead。硬左转 +60.7° 之后的下一段是 +67.5°。"),
    (209, 2): ("immediate_action", "confirmed", "写了 Go straight。进入餐厅前的第一段是 +69.5°。"),
    (1365, 3): ("immediate_action", "uncertain", "写了 Go straight。右转之后还有 -38.4° 和 -15.0°，同向累计约 53°。有直行这个词，但没有达到直接冲突所用的幅度。"),
    (364, 1): ("destination_description", "uncertain", "写沿走廊走到尽头，没有写 straight。起步已经朝向走廊，第一段却是 +61.5°。分不清这是走廊转弯，还是离开了眼前的走廊。"),
    (365, 1): ("destination_description", "uncertain", "只写走到走廊尽头。起步朝向走廊，第一段是 +61.5°。短语没有规定不许转弯，也没有证明路径离开了这条走廊。"),
    (366, 1): ("destination_description", "uncertain", "只写走到走廊尽头。情况和同路线另外两句一样，第一段是 +61.5°，不能据此确认直接冲突。"),
    (1731, 1): ("immediate_action", "confirmed", "写了 Go straight。第一段是 -92.3°，不是直行出门。"),
    (740, 1): ("destination_description", "not_confirmed", "只写沿走廊走。走廊在 -152.2° 之后，起步对着浴缸。先离开浴室再沿走廊走，和这句话相容。"),
    (1140, 1): ("destination_description", "not_confirmed", "写穿过走廊再穿过卧室，没有写 straight。起步在卧室里、走廊在 -170.6° 之后，只能说明短语没写起步转向，不能证明穿过走廊为假。"),
    (975, 1): ("immediate_action", "confirmed", "写离开浴室。起步画面是按摩床房间，不是浴室。冲突的是当前房间，不是目标在身后。"),
    (960, 1): ("immediate_action", "confirmed", "写离开卧室。起步画面在走廊里，不在卧室。"),
    (1342, 1): ("immediate_action", "uncertain", "短语规定在楼梯平台右转。路径后段有右转，被冻结标签反驳的是起步 +136.9°。这句话没有把这个左转规定成第一步。"),
    (1343, 1): ("destination_description", "not_confirmed", "只写下楼。楼梯不在起步正前方，第一段 +136.9°，随后高度下降。先转身再下楼，和这句话相容。"),
    (1351, 1): ("immediate_action", "confirmed", "写了 Walk straight down the hallway。起步已经朝向走廊，第一段是 +58.9°。"),
    (1352, 1): ("destination_description", "uncertain", "只写走到走廊远端，没有写 straight。起步朝向走廊，第一段是 +58.9°。不能确认这是离开走廊，还是走廊在这里转弯。"),
    (1360, 1): ("immediate_action", "confirmed", "写面对钟向左转。第一段是 -175.8° 右转。"),
    (1361, 1): ("immediate_action", "confirmed", "写了 Turn left。第一段是 -175.8° 右转。"),
    (1362, 2): ("immediate_action", "confirmed", "写了 go left。离开时第一段是 -175.8° 右转。"),
    (1531, 4): ("stop_condition", "confirmed", "停止条件是起居室门口。终点画面在房间内部，看着桌子和壁炉。"),
    (1532, 2): ("immediate_action", "confirmed", "写了 straight，并且走廊在左侧。记录的转弯是 -67.3° 和 -138.8°，都是向右。"),
    (1829, 3): ("stop_condition", "confirmed", "停止条件是门口。终点画面已经在房间内部看着沙发。"),
    (1116, 5): ("stop_condition", "confirmed", "停止条件是门外。终点画面已经在房间里面。"),
    (1493, 1): ("immediate_action", "confirmed", "写了 Walk all the way forward。红毯和门在 +175.5° 之后，不是一直向前。"),
    (81, 2): ("immediate_action", "confirmed", "写了 walk straight。进入走廊的第一段是 -106.7°。"),
    (1477, 2): ("immediate_action", "confirmed", "写了 walk straight。桌子和厨房在 -98.0° 之后，不是直行穿过。"),
    (1478, 1): ("destination_description", "not_confirmed", "只写经过桌子和椅子。它们在第一段 -98.0° 之后，短语没有规定直行。"),
    (1479, 1): ("immediate_action", "confirmed", "写了 straight towards，并从桌子左侧走。第一段是 -98.0°，不是直行。"),
    (1222, 1): ("destination_description", "not_confirmed", "只写走上棕色木台阶。台阶在 -82.0° 之后，起步对着草地。先转向台阶再走上去，和这句话相容。"),
    (1223, 1): ("immediate_action", "confirmed", "写了 Walk straight。第一段是 -82.0°，不是直行到门廊。"),
    (1224, 1): ("immediate_action", "confirmed", "写了 Walk forward。第一段是 -82.0°。"),
    (634, 1): ("immediate_action", "confirmed", "写了 Turn right。第一段是 +56.9° 左转。"),
}


def frozen_group(phrase: str, evidence: str) -> str:
    text = phrase.lower()
    if any(k in text for k in ("doorway", "just inside", "just outside", "outside of the room", "outside the room", "third step", "third stair", "stop in the door")):
        group = "stop_place"
    elif any(k in evidence for k in ("不是停在门口", "不是门口", "不是已经离开走廊", "仍在走廊", "门外")):
        group = "stop_place"
    elif any(k in evidence for k in ("不是左转", "不是右转", "不是向左", "不是向右", "不是右侧", "不是左侧", "画面左侧", "画面右侧", "没有右转", "没有左转")):
        group = "left_right"
    elif "指令要求" in evidence and ("左转" in evidence or "右转" in evidence):
        group = "left_right"
    elif "不是左侧走廊" in evidence or "不是右侧" in evidence:
        group = "left_right"
    elif any(k in text for k in ("walk up", "up the stair", "upstairs", "stairs down", "down the stair", "go up", "go down the")):
        group = "vertical"
    elif "dy " in evidence or "不是上楼" in evidence or "不是下楼" in evidence:
        group = "vertical"
    elif any(k in evidence for k in ("不是浴室", "不是从卧室", "不是掉头", "没有约 30", "没有这段右转")):
        group = "room_or_magnitude"
    else:
        group = "heading_after_turn"
    return group


def report_group(phrase: str, evidence: str, episode_id: str) -> str:
    group = frozen_group(phrase, evidence)
    if episode_id == "415" or episode_id in {"552", "1141", "76", "1477"}:
        return "heading_after_turn"
    if episode_id == "394":
        return "left_right"
    if group == "vertical" and episode_id != "686":
        return "heading_after_turn"
    if group == "vertical":
        return "vertical_opposite"
    return group


def main() -> None:
    phrases = list(csv.DictReader((ROOT / "random_audit_phrases.csv").open(encoding="utf-8")))
    contradicted = [row for row in phrases if row["verdict"] == "contradicted"]
    keys = {(int(row["episode_id"]), int(row["phrase_index"])) for row in contradicted}
    if keys != set(REVIEWS):
        missing = sorted(keys - set(REVIEWS))
        extra = sorted(set(REVIEWS) - keys)
        raise SystemExit(f"missing {missing[:12]} extra {extra[:12]}")

    out_rows = []
    for row in contradicted:
        key = (int(row["episode_id"]), int(row["phrase_index"]))
        constraint, status, note = REVIEWS[key]
        group = report_group(row["phrase"], row["evidence"], row["episode_id"])
        out_rows.append({
            "sample_index": row["sample_index"],
            "route_group_id": row["route_group_id"],
            "episode_id": row["episode_id"],
            "phrase_index": row["phrase_index"],
            "phrase": row["phrase"],
            "frozen_verdict": row["verdict"],
            "frozen_group": group,
            "instruction_constraint_type": constraint,
            "strict_conflict_status": status,
            "review_note": note,
        })

    counts = Counter(row["frozen_group"] for row in out_rows)
    expected = {
        "heading_after_turn": 92,
        "left_right": 16,
        "stop_place": 8,
        "room_or_magnitude": 5,
        "vertical_opposite": 1,
    }
    if dict(counts) != expected:
        raise SystemExit(f"group counts {dict(counts)} != {expected}")

    fields = list(out_rows[0])
    with (ROOT / "strict_conflict_review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    by_episode = defaultdict(list)
    for row in out_rows:
        by_episode[row["episode_id"]].append(row)
    instruction_rows = []
    for episode_id, rows in by_episode.items():
        statuses = {row["strict_conflict_status"] for row in rows}
        if "confirmed" in statuses:
            status = "confirmed"
        elif "uncertain" in statuses:
            status = "uncertain"
        else:
            status = "not_confirmed"
        instruction_rows.append({
            "sample_index": rows[0]["sample_index"],
            "route_group_id": rows[0]["route_group_id"],
            "episode_id": episode_id,
            "n_contradicted_phrases": len(rows),
            "strict_conflict_status": status,
            "constraint_types": " ".join(sorted({row["instruction_constraint_type"] for row in rows})),
            "phrase_indexes": " ".join(row["phrase_index"] for row in rows),
        })
    instruction_rows.sort(key=lambda row: (int(row["sample_index"]), int(row["episode_id"])))
    with (ROOT / "strict_conflict_review_instructions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(instruction_rows[0]))
        writer.writeheader()
        writer.writerows(instruction_rows)

    print("phrases", Counter(row["strict_conflict_status"] for row in out_rows))
    print("by group")
    for group in expected:
        subset = [row for row in out_rows if row["frozen_group"] == group]
        print(group, dict(Counter(row["strict_conflict_status"] for row in subset)))
    print("by type", dict(Counter(row["instruction_constraint_type"] for row in out_rows)))
    print("instructions", Counter(row["strict_conflict_status"] for row in instruction_rows), "n", len(instruction_rows))
    by_route = defaultdict(list)
    for row in out_rows:
        by_route[row["sample_index"]].append(row)
    route_status = Counter()
    for rows in by_route.values():
        statuses = {row["strict_conflict_status"] for row in rows}
        if "confirmed" in statuses:
            route_status["confirmed"] += 1
        elif "uncertain" in statuses:
            route_status["uncertain"] += 1
        else:
            route_status["not_confirmed"] += 1
    print("routes_with_contradicted", dict(route_status), "n", sum(route_status.values()))
    heading = [row for row in out_rows if row["frozen_group"] == "heading_after_turn"]
    print("heading92", dict(Counter(row["strict_conflict_status"] for row in heading)))
    print("heading92 types", dict(Counter((row["instruction_constraint_type"], row["strict_conflict_status"]) for row in heading)))


if __name__ == "__main__":
    main()
