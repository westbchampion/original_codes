# Generate the running commands
def gen_train_cmds():
	"""Generate commands to start a screen session and an interactive session to
	run the training."""
	runId = 57
	gpuId = 3
	homeDir = '/home/lhx'
	prjDir = f'{homeDir}/work/seg_gastric_cancer'
	logDir = f'{prjDir}/logs/gpu_1_{runId}'
	jobName = f"tr{runId:03}"

	cmds1 = [
		f"cd && rm -rvf {logDir} \\\n",
		"&& mkdir -p {lgd} \\\n&& cd {lgd}\n\n".format(lgd=logDir),
		f"screen -L -S {jobName}\n\n",
	]

	cmds2b = [
		f"export SINGULARITYENV_CUDA_VISIBLE_DEVICES={gpuId} \\\n",
	]

	cmds3b = [
		"&& singularity shell -C --nv \\\n",
		f"-B {homeDir}:{homeDir}:rw \\\n",
		f"{homeDir}/work/singularity_images/tensorflow_20.02-tf2-py3-lhx-20201016.sif\n\n",
	]

	cmds4 = [
		f"cd {prjDir} \\\n",
		f"&& python train.py \\\n",
		f"--config_file config/config_train.yaml\n",
	]

	cmds = cmds1 + cmds2b + cmds3b + cmds4
	with open('./cmds.txt', 'w') as f:
		f.writelines(cmds)
gen_train_cmds()
# %%













# %%
def gen_train_mrcnn_cmds():
	"""Generate commands to start a screen session and an interactive session to
	run the training."""
	runId = 2
	gpuId = 1
	cpuCount = 4
	homeDir = '/home/lhx'
	prjDir = f'{homeDir}/work/seg_gastric_cancer'
	screenDir = f'{homeDir}/work/screens'
	logDir = f'{screenDir}/{runId:03}'
	jobName = f"mrcnn{runId:03}"

	cmds1 = [
		f"cd && rm -rvf {logDir} \\\n",
		"&& mkdir -p {lgd} \\\n&& cd {lgd}\n\n".format(lgd=logDir),
		f"screen -L -S {jobName}\n\n",
	]

	cmds2b = [
		f"export SINGULARITYENV_CUDA_VISIBLE_DEVICES={gpuId} \\\n",
	]

	cmds3b = [
		"&& singularity shell -C --nv \\\n",
		f"-B {homeDir}:{homeDir}:rw \\\n",
		f"{homeDir}/work/singularity_images/tensorflow_19.12-tf1-py3-mrcnn-20201016.sif\n\n",
	]

	cmds4 = [
		f"cd {prjDir} \\\n",
		f"&& python mrcnn_seg.py train \\\n",
		f"--dataset {prjDir}/data/train \\\n",
		f"--weights coco \\\n",
		f"--cpuCount {cpuCount} \\\n",
		f"--imgLen {512} \\\n",
		f"--logs {logDir}\n",
	]

	cmds = cmds1 + cmds2b + cmds3b + cmds4
	with open('./cmds.txt', 'w') as f:
		f.writelines(cmds)
gen_train_mrcnn_cmds()
# %%





# %%