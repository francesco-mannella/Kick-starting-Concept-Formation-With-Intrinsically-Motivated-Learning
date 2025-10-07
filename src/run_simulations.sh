for i in 0 1 2 3 4 5 6 7 8 9; do
  echo "python SMMain.py -s 100$i -w -t 50000 --parasite --group unsupervised_touch_refact --name unsupervised_touch_refact_s100$i" | batch
done;
