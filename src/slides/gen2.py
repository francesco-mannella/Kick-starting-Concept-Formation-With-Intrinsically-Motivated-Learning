import json, os, sys
M = 1_000_000
s = sys.argv[1]
CFG = {"45": dict(page="h39bceb20c3d3c807_0_0", demo="1K-Uzm5vr9qIVQ3SW4Mgshuyx0ssGNtyL", epi="16a1lb_izBQ308k0kZpt63RbzffcG5UeI", epi_ar=720 / 871,
                  frames={"fov": "h39bceb20c3d3c807_0_6", "pro": "h39bceb20c3d3c807_0_13", "som": "h39bceb20c3d3c807_0_16"},
                  txt=dict(obj="h39bceb20c3d3c807_0_11", goal="h39bceb20c3d3c807_0_10", start="h39bceb20c3d3c807_0_19", sens="h39bceb20c3d3c807_0_20", end="h39bceb20c3d3c807_0_21")),
       "46": dict(page="h39bceb20c3d3c807_0_24", demo="1dZsf-6hvxQ3eDxnY4IE8F7wptErJY946", epi="1aHfUuGD8s9Y8bBBCz15zCttugCyYLmE_", epi_ar=720 / 868,
                  frames={"pro": "h39bceb20c3d3c807_0_38", "som": "h39bceb20c3d3c807_0_41", "fov": "h39bceb20c3d3c807_0_44"},
                  txt=dict(obj="h39bceb20c3d3c807_0_32", goal="h39bceb20c3d3c807_0_29", start="h39bceb20c3d3c807_0_34", sens="h39bceb20c3d3c807_0_35", end="h39bceb20c3d3c807_0_36")),
       "47": dict(page="h39bceb20c3d3c807_0_48", demo="1XKQNl7SMK8d63AyvwZBNvXIOvGguxbcK", epi="1ZBeTHHy4QCy63n7RkhJ44oSSv06bwHmK", epi_ar=720 / 853,
                  frames={"fov": "h39bceb20c3d3c807_0_56", "som": "h39bceb20c3d3c807_0_61", "pro": "h39bceb20c3d3c807_0_62"},
                  txt=dict(obj="h39bceb20c3d3c807_0_59", goal="h39bceb20c3d3c807_0_58", start="h39bceb20c3d3c807_0_63", sens="h39bceb20c3d3c807_0_64", end="h39bceb20c3d3c807_0_66"))}[s]
COL = {"fov": 5.05 * M, "som": 6.70 * M, "pro": 8.35 * M}; F = 1.0 * M; FY = 2.72 * M
REFRESH = os.environ.get("REFRESH") == "1"; OLD = os.environ.get("PREFIX_OLD", "kss"); NEW = os.environ.get("PREFIX_NEW", "k2")
CFG["demo"] = os.environ.get("DEMO_ID", CFG["demo"]); CFG["epi"] = os.environ.get("EPI_ID", CFG["epi"])
plan = json.load(open('plan2.json'))[s]; P = CFG["page"]; R = []
url = lambda i: f"https://drive.google.com/uc?export=view&id={i}"
tf = lambda x, y, sx=1, sy=1: {"scaleX": sx, "scaleY": sy, "translateX": int(x), "translateY": int(y), "unit": "EMU"}
for it in plan: R.append({"deleteObject": {"objectId": f"{OLD}{s}_{it['kind']}"}})
if REFRESH: R.append({"deleteObject": {"objectId": f"{OLD}{s}_episode"}})
for it in plan:
    oid = f"{NEW}{s}_{it['kind']}"
    R.append({"createImage": {"objectId": oid, "url": url(CFG["demo"]), "elementProperties": {"pageObjectId": P, "size": {"width": {"magnitude": 40000, "unit": "EMU"}, "height": {"magnitude": 53325, "unit": "EMU"}}, "transform": tf(it["x"], it["y"])}}})
    R.append({"updateImageProperties": {"objectId": oid, "imageProperties": {"cropProperties": {"leftOffset": round(it["L"], 5), "rightOffset": round(it["R"], 5), "topOffset": round(it["T"], 5), "bottomOffset": round(it["B"], 5)}},
              "fields": "cropProperties.leftOffset,cropProperties.rightOffset,cropProperties.topOffset,cropProperties.bottomOffset"}})
    R.append({"updatePageElementTransform": {"objectId": oid, "applyMode": "ABSOLUTE", "transform": tf(it["x"], it["y"], it["w"] / 40000, it["h"] / 53325)}})
# episode gif, fit in box
bx, by, bw, bh = 4.5 * M, 0.55 * M, 1.6 * M, 1.65 * M; ar = CFG["epi_ar"]; w = min(bw, bh * ar); h = w / ar
R.append({"createImage": {"objectId": f"{NEW}{s}_episode", "url": url(CFG["epi"]), "elementProperties": {"pageObjectId": P, "size": {"width": {"magnitude": int(w), "unit": "EMU"}, "height": {"magnitude": int(h), "unit": "EMU"}},
          "transform": tf(bx + (bw - w) / 2, by + (bh - h) / 2)}}})
for k, fid in ([] if REFRESH else CFG["frames"].items()): R.append({"updatePageElementTransform": {"objectId": fid, "applyMode": "ABSOLUTE", "transform": tf(COL[k] - F / 2, FY, F / 3e6, F / 3e6)}})
T = CFG["txt"]
R += [] if REFRESH else [{"updatePageElementTransform": {"objectId": T["start"], "applyMode": "ABSOLUTE", "transform": tf(0.555 * M, 1.58 * M, 0.1803, 0.1231)}},
      {"updatePageElementTransform": {"objectId": T["end"], "applyMode": "ABSOLUTE", "transform": tf(1.755 * M, 1.58 * M, 0.1803, 0.1231)}},
      {"updatePageElementTransform": {"objectId": T["sens"], "applyMode": "ABSOLUTE", "transform": tf(0.64 * M, 1.93 * M, 0.5225, 0.1231)}},
      {"updatePageElementTransform": {"objectId": T["goal"], "applyMode": "ABSOLUTE", "transform": tf(6.35 * M, 2.0 * M, 0.2942, 0.1539)}},
      {"updatePageElementTransform": {"objectId": T["obj"], "applyMode": "ABSOLUTE", "transform": tf(6.45 * M, 0.30 * M, 0.8667, 0.431)}},
      {"updateTextStyle": {"objectId": T["obj"], "textRange": {"type": "ALL"}, "style": {"fontSize": {"magnitude": 16, "unit": "PT"}}, "fields": "fontSize"}}]
for k, name in (() if REFRESH else (("fov", "Foveal vision"), ("som", "Somatosensory"), ("pro", "Proprioception"))):
    oid = f"k2{s}_title_{k}"
    R += [{"createShape": {"objectId": oid, "shapeType": "TEXT_BOX", "elementProperties": {"pageObjectId": P, "size": {"width": {"magnitude": 1500000, "unit": "EMU"}, "height": {"magnitude": 300000, "unit": "EMU"}}, "transform": tf(COL[k] - 0.75 * M, 2.40 * M)}}},
          {"insertText": {"objectId": oid, "text": name}},
          {"updateTextStyle": {"objectId": oid, "textRange": {"type": "ALL"}, "style": {"fontFamily": "Arial", "fontSize": {"magnitude": 11, "unit": "PT"}, "foregroundColor": {"opaqueColor": {"themeColor": "DARK2"}}}, "fields": "fontFamily,fontSize,foregroundColor"}},
          {"updateParagraphStyle": {"objectId": oid, "textRange": {"type": "ALL"}, "style": {"alignment": "CENTER"}, "fields": "alignment"}}]
print(json.dumps(R, separators=(',', ':')))
