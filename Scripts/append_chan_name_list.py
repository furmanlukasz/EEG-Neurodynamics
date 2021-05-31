with open('channel_names.txt', 'r') as f:
  score_list = f.readline()
  print(score_list)

Eall = list(open("channel_names.txt"))
names = [Eall[i][1:3] for i in range(len(Eall))]
names
with open("Scripts/channel_names.txt", "r") as backfile:
    lines = backfile.readlines()
    lines
    len(lines)