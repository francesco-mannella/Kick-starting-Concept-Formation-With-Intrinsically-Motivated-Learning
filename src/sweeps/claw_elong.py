import sys, pickle, numpy as np
sys.argv=[sys.argv[0], sys.argv[1]]
exec(open("/home/fmannella/tmp/kss/claw_analyze.py").read().split("A = [k for k in cells")[0])
A=[k for k in cells if k[0]=="A"]; B=[k for k in cells if k[0]=="B"]
print("elongated shapes only (stick, bar, plate):")
decode([[k for k in A if k[1:4]==(sh,sc,0.2)] for sh in ("stick","bar","plate") for sc in (0.6,1.0,1.6)],"ROTATION 0/45/90, same shape and size","0.33")
decode([[k for k in B if k[1]==sh and k[4]==pos] for sh in ("stick","bar","plate") for pos in sorted(set(k[4] for k in B))],"ROTATION 0/90, same shape and position","0.50")
print("compact shapes only:")
decode([[k for k in A if k[1:4]==(sh,sc,0.2)] for sh in ("cube1","cube2","disk3","disk5","hex4","wedge") for sc in (0.6,1.0,1.6)],"ROTATION 0/45/90, same shape and size","0.33")
