def extract_lines(lines, prefix):
    extracted_lines = []
    previous_lines = []
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            extracted_lines.append(line.strip())
            if i > 0:
                previous_lines.append(lines[i-1].strip())
            else:
                previous_lines.append("")  # No previous line for the first line
    return extracted_lines, previous_lines

def calc_total_time(previous_lines):
    import re
    total_seconds = 0
    for prev_line in previous_lines:
        # Extract time in format [MM:SS<...] from the line
        match = re.search(r'\[(\d+):(\d+)<', prev_line)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            total_seconds += minutes * 60 + seconds

    remaining_seconds = total_seconds % 60
    total_hours = total_seconds // 3600
    remaining_minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    return total_hours, remaining_minutes, remaining_seconds

def calc_mean_time_per_epoch(previous_lines):
    import re
    total_seconds = 0
    count = 0
    for prev_line in previous_lines:
        # Extract time in format [MM:SS<...] from the line
        match = re.search(r'\[(\d+):(\d+)<', prev_line)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            total_seconds += minutes * 60 + seconds
            count += 1

    if count == 0:
        return 0, 0, 0  # Avoid division by zero

    mean_seconds = total_seconds // count
    remaining_seconds = mean_seconds % 60
    total_hours = mean_seconds // 3600
    remaining_minutes = (mean_seconds % 3600) // 60
    remaining_seconds = mean_seconds % 60
    return total_hours, remaining_minutes, remaining_seconds

################################# PLOT FILE 1 #######################################################

path = "training_log_x3.txt"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# extract lines that start with "eval psnr"
#eval_psnr_lines = [line.strip() for line in lines if line.startswith("eval psnr")]
eval_psnr_lines, previous_lines = extract_lines(lines, "eval psnr")
total_hours, remaining_minutes, remaining_seconds = calc_total_time(previous_lines)
print(f"Total training time for scale 3: {total_hours}:{remaining_minutes:02d}:{remaining_seconds:02d}")

#plot epoch vs psnr
import matplotlib.pyplot as plt
epochs = list(range(len(eval_psnr_lines)))
psnrs = [float(line.split(":")[1].strip()) for line in eval_psnr_lines]
plt.plot(epochs, psnrs)
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.title("Scale 3: PSNR vs Epoch")
plt.grid()
plt.savefig("x3_psnr_vs_epoch.png")
plt.close()

#Print Peak PSNR
peak_psnr = max(psnrs)
print("Peak PSNR for scale 3: " + str(peak_psnr))


################################# PLOT FILE 2 #######################################################

path = "training_log_x2.txt"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# extract lines that start with "eval psnr"
eval_psnr_lines, previous_lines = extract_lines(lines, "eval psnr")
total_hours, remaining_minutes, remaining_seconds = calc_total_time(previous_lines)
print(f"Total training time for scale 2: {total_hours}:{remaining_minutes:02d}:{remaining_seconds:02d}")



#plot epoch vs psnr
import matplotlib.pyplot as plt
epochs = list(range(len(eval_psnr_lines)))
psnrs = [float(line.split(":")[1].strip()) for line in eval_psnr_lines]
plt.plot(epochs, psnrs)
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.title("Scale 2: PSNR vs Epoch")
plt.grid()
plt.savefig("x2_psnr_vs_epoch.png")
plt.close()

#Print Peak PSNR
peak_psnr = max(psnrs)
print("Peak PSNR for scale 2: " + str(peak_psnr))

################################# PLOT FILE 3 #######################################################

path = "training_log_x2_200e.txt"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# extract lines that start with "eval psnr"
eval_psnr_lines, previous_lines = extract_lines(lines, "eval psnr")
total_hours, remaining_minutes, remaining_seconds = calc_total_time(previous_lines)
print(f"Total training time for 200 epochs scale 2: {total_hours}:{remaining_minutes:02d}:{remaining_seconds:02d}")



#plot epoch vs psnr
import matplotlib.pyplot as plt
epochs = list(range(len(eval_psnr_lines)))
psnrs = [float(line.split(":")[1].strip()) for line in eval_psnr_lines]
plt.plot(epochs, psnrs)
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.title("Scale 2: PSNR vs Epoch")
plt.grid()
plt.savefig("200_x2_psnr_vs_epoch.png")

#Print Peak PSNR
peak_psnr = max(psnrs)
print("Peak PSNR for 200epochs scale 2: " + str(peak_psnr))

