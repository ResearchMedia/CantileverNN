'''
@Title: RandomBeamOptimizer
@Author: Philippe Wyder (pmw2125)
@Description: performs random search for a high-performing beam given one or multiple criteria
@ToDo: the memory footprint is relatively high: evaluate if variables copies can be avoided through smarter program flow
'''
import torch
#import importlib
import math, random, argparse, copy
import numpy as np
from PIL import Image, ImageDraw
from skimage import io, transform
from skimage.color import rgb2gray
from skimage.io._plugins.pil_plugin import *
from pathlib import Path
from queue import LifoQueue
from collections import OrderedDict 
import time, csv, json
import torch.multiprocessing as mp
from NNEvaluator import NNEvaluator
DEBUG_MODE = False
write_plot_file = False
WRITE_OPTIMIZATION_LOG = False
PRINT_OPTIMIZATION_PROGRESS = False

result_path = Path('/home/ron/Downloads/optimized_beam_test')
#result_path = Path('/data/pmw2125/optimized_beam_test')
plot_file_ext = '.optim_plot'
plot_file_folder = Path('tmp/')
minRad = 24
maxRad = 63
irregularity = 0.4
spikeyness = 0.1
minNumVerts = 3
maxNumVerts = 30
img_res = 128
img_center = img_res/2
mypool = None
# Move parameters
step_size_max = 2.5
MAX_MULTI_MOVE = 2
ALLOW_VERT_REMOVAL = False
# Smoothing Parameters
MAX_SPIKEYNESS = 5.5
ACUTE_ANGLE_THRESHOLD = np.radians(15)

# Evolution Parameters:
POPULATION_SIZE = 100
POP_SPLIT = 0.5
BIRTH_RATE = 0.1

# Run Time String (Ensures consistent file naming)
time_string = time.strftime('%Y%m%d_%H-%M', time.localtime(time.time()))

'''
Optimizer with parallelized multi-evaluator approach
'''
def findConfiguration(job_params, eval_config, num_results = 1):
	# inputs
	#	job_params = (mode, desired_result, max_iter, loss_function, maxNumVerts, ALLOW_VERT_REMOVAL)	
	#	eval_config = {'GPUs', 'workers_per_gpu', 'model_name', 'NN', 'num_classes', 'img_res'}
	#	num_results = nr_runs

	if isinstance(job_params, tuple) and job_params[0] == 'E':
		fitness = []
		for gpu_id in eval_config['GPUs']:
			for instance in range(eval_config['workers_per_gpu']):
				fitness.append(NNEvaluator(eval_config['model_name'], gpu_id, eval_config['NN'], 
											num_classes=eval_config['num_classes'], img_res = eval_config['img_res']))

		max_iter = int(job_params[2]/POPULATION_SIZE) # for fair comparison
		results = _evolution(job_params[1], fitness, max_iter, job_params[3], 
								population_size = POPULATION_SIZE, pop_split = POP_SPLIT, birth_rate = BIRTH_RATE,
								allowVertRemoval = job_params[5])
	elif isinstance(job_params, tuple) and num_results == 1:
		evaluator = NNEvaluator(eval_config['model_name'], eval_config['GPUs'][0], eval_config['NN'], 
									num_classes=eval_config['num_classes'], img_res = eval_config['img_res'])
		if job_params[0] == 'H':
			results = _hillClimberSingleMove(evaluator, job_params[1:])
		if job_params[0] == 'R':
			results = _randomSearch(evaluator, job_params[1:])
	else:
		results = parallelizedOptimization(eval_config, num_results, job_params)

	return results
def parallelizedOptimization(eval_config, nr_itterations, job_params):
	ctx = mp.get_context("spawn")
	output_q = ctx.Queue(nr_itterations)
	nr_desired_results = 1
	# fill jobs Queue
	if isinstance(job_params, tuple):
		jobs_q = ctx.Queue(nr_itterations)
		print("Received Single set of desired_results: ", job_params[1])
		for i in range(nr_itterations):
			jobs_q.put([i, job_params])
	else:
		nr_desired_results = len(job_params)
		jobs_q = ctx.Queue(nr_itterations*nr_desired_results)
		#print("Received {} desired_results".format(nr_desired_results))
		job_counter = 0
		for param_set in job_params:
			for i in range(nr_itterations):
				jobs_q.put([job_counter, param_set])
				job_counter += 1

	# launch processes
	processes = []
	worker_id = 0
	for gpu_id in eval_config['GPUs']:
		print("starting workers on GPU{} ".format(gpu_id))
		for i in range(eval_config['workers_per_gpu']):
			p = ctx.Process(target=optimizationWorker, args=(worker_id, gpu_id, eval_config, jobs_q, output_q))
			processes.append(p)
			p.start()
			worker_id += 1

	print("Started {} jobs on {} workers with {} desired_result configurations...".format(
			nr_itterations*nr_desired_results, len(processes), nr_desired_results))

	# collect results from output_q
	result_list = []
	while len(result_list) < nr_itterations*nr_desired_results:
		if output_q.empty():
			pass
		else:
			result_list.append(output_q.get())
	return result_list
		
def optimizationWorker(worker_id, gpu_id, eval_config, jobs_q, output_q):
	print("Worker #{} is started on GPU:{}".format(worker_id, gpu_id))

	evaluator =	NNEvaluator(eval_config['model_name'], gpu_id, eval_config['NN'], 
				num_classes=eval_config['num_classes'], img_res = eval_config['img_res'])
	while True:
		if jobs_q.empty():
			print("jobs_q was empty - worker #{} is returning".format(worker_id))
			break
		else:
			job_id, cur_params = jobs_q.get()
			print("I am worker #{} running on GPU:{} doing job #{}".format(worker_id, gpu_id, job_id))
			if cur_params[0] == 'H':
				output_q.put(_hillClimberSingleMove(evaluator, cur_params[1:]))
			if cur_params[0] == 'R':
				output_q.put(_randomSearch(evaluator, cur_params[1:]))

def _randomSearch(fitness, params):
	# assign parameter values from arguments
	desired_result, max_iter, loss_function, numVerts, allowVertRemoval = params	

	start_time = time.time()

	best_beam_history = []
	best_beam = {}

	# generate initial beam
	aveRadius = random.randint(minRad,maxRad)
	numVerts = random.randint(minNumVerts,maxNumVerts) 
	best_beam['verts'] = generatePolygon(img_center,img_center, aveRadius, irregularity, spikeyness, numVerts )	
	# DEBUG CODE
	if DEBUG_MODE:
		from IPython import embed; embed()
	# evaluate initial beam beam
	best_beam['fitness'], best_beam['nrpx'] = _getEfFitness(best_beam['verts'], desired_result, fitness, mode = loss_function)
	best_beam_history.append(copy.deepcopy(best_beam))

	# current beam == best beam for first iteration
	cur_fitness = best_beam['fitness']
	cur_nrpx = best_beam['nrpx']
	cur_verts = best_beam['verts']
	for i in range(int(max_iter)):
		if ( cur_fitness < best_beam['fitness'] ):
			best_beam['fitness'] = cur_fitness
			best_beam['nrpx'] = cur_nrpx
			best_beam['verts'] = cur_verts
			if PRINT_OPTIMIZATION_PROGRESS: print("{} - Best Fitness: {}".format(i, best_beam['fitness']))
			if write_plot_file:
				best_verts_history = [c['verts'] for c in best_beam_history]
				_writePlotFile(best_verts_history, "R")
		if WRITE_OPTIMIZATION_LOG:		
			_writeToOptimizationLog(i, best_beam['fitness'], desired_result, time_string, 'randomSearch')
		if i%1000 == 0 and PRINT_OPTIMIZATION_PROGRESS:
			print("Tried {} random beams in {} (last fitness: {})".format(i, time.time()-start_time, cur_fitness))
		# generate & evaluate sample for next iteration
		aveRadius = random.randint(minRad, maxRad)
		numVerts = random.randint(minNumVerts,maxNumVerts)
		cur_verts = generatePolygon(img_center,img_center, aveRadius, irregularity, spikeyness, numVerts ) 
		cur_fitness, cur_nrpx = _getEfFitness(cur_verts, desired_result, fitness, mode = loss_function)
	return best_beam['verts'], best_beam['fitness'], desired_result

def _hillClimberSingleMove(fitness, params):
	# assign parameter values from arguments
	desired_result, max_iter, loss_function, numVerts, allowVertRemoval = params
	start_time = time.time()

	aveRadius = random.randint(minRad,maxRad)
	#numVerts = random.randint(minNumVerts,maxNumVerts) 

	best_beam_history = []
	best_verts_history = []
	# Initialize first beam
	best_beam = {}
	best_beam['verts'] = generatePolygon(img_center,img_center, aveRadius, irregularity, spikeyness, numVerts )	
	if DEBUG_MODE:
		from IPython import embed; embed()
	# evaluate first beam
	best_beam['fitness'], best_beam['nrpx'] = _getEfFitness(best_beam['verts'], desired_result, fitness, mode = loss_function)
	cur_fitness = best_beam['fitness']	
	# first beam == current beam
	cur_verts = best_beam['verts']
	cur_fitness = best_beam['fitness']
	cur_nrpx = best_beam['nrpx']
	for i in range(int(max_iter)):
		cur_fitness, cur_nrpx = _getEfFitness(cur_verts, desired_result, fitness, mode = loss_function)
		if(cur_fitness <=  best_beam['fitness'] 
			and cur_verts not in best_verts_history and cur_nrpx > 700 and isNonIntersecting(cur_verts)):
			best_beam['fitness'] = cur_fitness
			best_beam['nrpx'] = cur_nrpx
			best_beam['verts'] = cur_verts
			best_beam_history.append(copy.deepcopy(best_beam))
			best_verts_history.append(copy.deepcopy(best_beam['verts']))
			if PRINT_OPTIMIZATION_PROGRESS: print("{} - Best Fitness: {}".format(i, best_beam['fitness']))
			if write_plot_file:
				_writePlotFile(best_verts_history, "H")
		if WRITE_OPTIMIZATION_LOG:
			_writeToOptimizationLog(i, best_beam['fitness'], desired_result, time_string, 'HillClimber')
		if i%1000 == 0 and PRINT_OPTIMIZATION_PROGRESS:
			print("Tried {} beams in {} (last fitness: {})".format(i, time.time()-start_time, cur_fitness))
		# Take Step
		cur_verts = _HCMove(best_beam['verts'], step_size_max)
		# delete duplicate vertices # may FUP order
		cur_verts = removeDuplicates(cur_verts)

		# If Acute Angle Threshold is set (REMOVE ACUTE ANGLES)
		if ACUTE_ANGLE_THRESHOLD:
			cur_verts = removeAcuteAngle(cur_verts, ACUTE_ANGLE_THRESHOLD)
		# Remove Quasy Colinear Vertices if allowVertRemoval == True
		if allowVertRemoval:
			cur_verts = removeQuasiColinearVertices(cur_verts, eta=1e-4)
		# smoothen spikey shapes to avoid impossible cross-sections
		while _getSpikeyness(cur_verts) > MAX_SPIKEYNESS:
			#print("Too Spikey-smoothing", _getSpikeyness(cur_verts))
			cur_verts = _smoothenSpikeyness(cur_verts)	
	return best_beam['verts'], best_beam['fitness'], desired_result

def _HCMove(verts, step_size_max):
	#if (random.random() > 0.3):
	#cur_verts = _MultiMove(verts, MAX_MULTI_MOVE)
	cur_verts = _SingleMove(verts, step_size_max)
	#else:
	#	cur_verts = _growShrinkShape(verts, max_change = 0.2)
	return cur_verts

def _SingleMove(verts, step_size_max):
	''' Moves one vertex and resorts the vertices by angle to avoid self-intersection'''
	i = random.randint(0,len(verts)-1)
	step_size_max_theta = 4*np.pi/len(verts) 
	new_verts = convert2Polar(verts, img_center, img_center)
	step_size_rad = random.random() * step_size_max
	step_size_theta = random.random() * step_size_max_theta
	add_subtract_rad = random.choice([-1,0,1])	
	add_subtract_theta = random.choice([-1,0,1])	
	cur_rad = new_verts[i][0] + add_subtract_rad * step_size_rad
	cur_theta = new_verts[i][1] + add_subtract_theta * step_size_theta
	# ensure new vertex is not exceeding max/min radius constraints
	if cur_rad > maxRad:
		cur_rad = maxRad
	elif cur_rad < minRad:
		cur_rad = minRad
	new_verts[i] = (cur_rad, cur_theta)
	# sort vertices by angle to avoid self-overlapping shapes:
	new_verts = sorted(new_verts, key = lambda v: v[1])
	# convertFromPolar handles vertices exceeding 2*pi
	cur_verts = convertFromPolar(new_verts, img_center, img_center)
	return cur_verts
def _MultiMove(verts, step_size_max):
	''' Moves one vertex and resorts the vertices by angle to avoid self-intersection'''
	np_verts = np.array(verts)
	# random normal distribution
	steps = np.random.randn(*np_verts.shape)*step_size_max
	np_verts = np_verts + steps
	new_verts = convert2Polar(np_verts, img_center, img_center)
	# ensure new vertex is not exceeding max/min radius constraints
	for i in range(len(new_verts)):
		cur_rad = new_verts[i][0]
		cur_theta = new_verts[i][1]
		if cur_rad > maxRad:
			cur_rad = maxRad
		elif cur_rad < minRad:
			cur_rad = minRad
		new_verts[i] = (cur_rad, cur_theta)
	# sort vertices by angle to avoid self-overlapping shapes:
	new_verts = sorted(new_verts, key = lambda v: v[1])
	# convertFromPolar handles vertices exceeding 2*pi
	cur_verts = convertFromPolar(new_verts, img_center, img_center)
	return cur_verts
# grow/shrink an existing shape
def _growShrinkShape(verts, max_change=0.2):
	p_verts = np.array(convert2Polar(verts, img_center, img_center))
	scale_factor = 1 + random.random() * max_change*2 - max_change
	p_verts[:,0] = p_verts[:,0]*scale_factor
	return convertFromPolar(p_verts, img_center, img_center)	

'''
Evolution
Step One: Generate the initial population of individuals randomly. (First generation)

Step Two: Evaluate the fitness of each individual in that population (time limit, sufficient fitness achieved, etc.)

Step Three: Repeat the following regenerational steps until termination:

	Select the best-fit individuals for reproduction. (Parents)
	Breed new individuals through crossover and mutation operations to give birth to offspring.
	Evaluate the individual fitness of new individuals.
	Replace least-fit population with new individuals.
'''
def _evolution(desired_result, fitness, max_iter, loss_function = 'MAPE',
				 population_size = 100, pop_split = 0.5, birth_rate = 0.2, allowVertRemoval = False):
	plot_selection = 9
	start_time = time.time()
	assert pop_split > birth_rate
	split_idx = int(population_size*pop_split)
	incarnation_idx = int(population_size*birth_rate)
	# Get starting population	
	population = _getPopulation(population_size)
	if DEBUG_MODE:
		from IPython import embed; embed()
	population =_evaluatePopulation(population, desired_result, fitness, loss_function)	
	population = sorted(population, key = lambda p: p['fitness'])
	for i in range(max_iter):
		# Print Epoch performance
		if PRINT_OPTIMIZATION_PROGRESS: print("Evolution epoch: {}, runtime: {}, best_fitness: {}".format(
												i, time.time()-start_time, population[0]['fitness']))
		# write optimization log
		if WRITE_OPTIMIZATION_LOG:
			for idx, p in enumerate(population):
				_writeToOptimizationLog(i*population_size+idx, population[0]['fitness'], desired_result, time_string, 'evolution')
		# Age the population
		_WriteEvolutionLog(i, population, desired_result = desired_result, time_string = time_string)
		# plot top 10 solutions
		for j in range(plot_selection):
			if _shapeIsIllegal(population[j]['verts']):
				print("Illegal Shape In Population with fitness", population[j]['fitness'])
			if write_plot_file:
				_writePlotFile([population[j]['verts']], str(j))

		parents = _selectAgePareto(population, split_idx)	
		if PRINT_OPTIMIZATION_PROGRESS:
			for p in parents[:10]: print("parents fitness {} with age {}".format(p['fitness'], p['age']))

		children = _breed(parents)
		parents  = _movePopulation(parents, step_size_max)
		children  = _movePopulation(children, step_size_max, allowVertRemoval = True)

		# remove a subset of parents in favour of new random samples
		new_incarnations = _getPopulation(incarnation_idx)
		newFamilies = parents[:-incarnation_idx] + children + new_incarnations
		newFamilies = _evaluatePopulation(newFamilies, desired_result, fitness, loss_function)

		# combine the newly bred/morphed/generated samples with the past population
		newPopulation = newFamilies + population
		# pareto select the new population
		newPopulation = sorted(newPopulation, key = lambda p: p['fitness'])

		population = _selectAgePareto(newPopulation, population_size)
		assert len(population) == (population_size)

		# increment age for all elements in the population
		for idx, p in enumerate(population):
			p['age'] += 1 # increment age of each datapoint
		
	# return population with MAPE error measure for more clarity
	population = _evaluatePopulation(population, desired_result, fitness, 'MAPE')
	return [(p['verts'], p['fitness'], desired_result) for p in population]
def _getPopulation(population_size):
	pop = []
	for i in range(population_size):
		beam = _getNewBeam()
		pop.append(copy.deepcopy(beam))
	return pop
def _getNewBeam():
	beam = {}
	aveRadius = random.randint(minRad, maxRad)
	numVerts = random.randint(minNumVerts, maxNumVerts) 
	beam['verts'] = generatePolygon(img_center, img_center, aveRadius, irregularity, spikeyness, numVerts )	
	beam['age'] = 1
	beam['fitness'] = None
	beam['nrpx'] = None
	return beam
# Age pareto selection
def _getAgeParetoFront(sorted_population):
	population = copy.deepcopy(sorted_population)
	for i, p in enumerate(sorted_population):
		for j, q in enumerate(sorted_population):
			if 	((p['age'] > q['age'] and p['fitness'] >= q['fitness']) or  
				(not i == j and p['age'] == q['age'] and p['fitness'] > q['fitness'])):
				population.remove(p)
				#print("Removed {} fit - {} age \n because of {} fit - {} age".format(
				#	p['fitness'], p['age'], q['fitness'], q['age']))
				break
	return population
def _selectAgePareto(sorted_population, nr_parents):
	parents = []
	#fronts = []
	population = copy.deepcopy(sorted_population)
	while len(parents) < nr_parents:
		population = [p for p in population if p not in parents]
		pareto_front = _getAgeParetoFront(population)
		parents.extend(sorted(pareto_front, key = lambda p: p['fitness']))
		#fronts.append(pareto_front)
		#print("Select_agePareto: len(parents):", len(parents), "\t len(sorted_population)", len(population),
		#		"nr_parents", nr_parents)
	return parents[:nr_parents]
def _selectParentsByFitness(sorted_population, nr_parents, rounding_level=4):
	pop = sorted_population # i.e. sorted best to worst
	parents = {}
	i = 0
	while(len(parents) < nr_parents and i < len(pop)):
		cur_fit = str(round(pop[i]['fitness'], rounding_level))
		if cur_fit not in parents:
			parents[cur_fit] = copy.deepcopy(pop[i])
		#print("Want {} parents, got {}".format(nr_parents, len(parents)))
		i += 1
	parents_list = [parents[p] for p in parents] 
	while not len(parents_list) == nr_parents:
		parents_list.append(_getNewBeam())
	return parents_list
def _selectParentsByOverlap(sorted_population, nr_parents, threshold = 0.01):
	pop = sorted_population # i.e. sorted best to worst
	parents = []
	parents.append(copy.deepcopy(pop[0]))
	for i in range(1,len(pop)):	
		if len(parents) < nr_parents:
			elligible_parent = True
			jobs_list = [(p, pop[i], threshold) for p in parents]
			elligibility_list = mypool.starmap(_isElligibleByOverlap, jobs_list)
			'''
			for p in parents:
				if threshold > _getSimilarityOfShape(pop[i], p):
					print("too similar", _getSimilarityOfShape(pop[i], p))
					elligible_parent = False
					break
			'''
			# if vertices are different enough
			if (False not in elligibility_list):
				parents.append(copy.deepcopy(pop[i]))
			#print("[{}]\tWant {} parents, got {}".format(i,nr_parents, len(parents)))
		else:
			break
	# fill missing parents with random samples
	while not len(parents) == nr_parents:
		parents.append(_getNewBeam())
	return parents
def _isElligibleByOverlap(existing_parent, candidate, threshold):
	return threshold < _getSimilarityOfShape(existing_parent, candidate)

def _movePopulation(population, step_size_max, allowVertRemoval = False):
	for p in population:
		#if random.random() > 0.3:
		p['verts'] = _SingleMove(p['verts'], step_size_max)
		#p['verts'] = _MultiMove(p['verts'], MAX_MULTI_MOVE)
		#else:
		#	p['verts'] = _growShrinkShape(p['verts'], max_change = 0.1)
		#if _getSpikeyness(p['verts']) > MAX_SPIKEYNESS:
		# 	p['verts'] = _smoothenSpikeyness(p['verts'])
		#from IPython import embed; embed()
		if allowVertRemoval:
			p['verts'] = removeQuasiColinearVertices(p['verts'], eta=1e-3)

	return population
def _breed(parents):
	children = []
	for i in range(len(parents)):
		j = random.randint(0,len(parents)-1)
		children.append(
			_sectorMeanCrossover(parents[i], parents[j], img_center, img_center))
			#_sliceCrossover(parents[i], parents[j], img_center, img_center)) 
	return children
def _evaluatePopulation(population, desired_result, fitness, loss_function):
	nr_workers = len(fitness)
	jobs_list = []
	sub_populations = _splitWorkload(population, nr_workers)
	#print("length of population:", len(population))
	#print("length of sub-populations:", len(sub_populations))
	#print("length of fitness:", len(fitness))
	for fit_f, pop in zip(fitness, sub_populations):
		jobs_list.append((pop, desired_result, fit_f, loss_function))
	#sub_populations = mypool.starmap(_evaluatePopulationWorker, jobs_list)
	sub_populations = [_evaluatePopulationWorker(*arg_tuple) for arg_tuple in jobs_list]
	return [copy.deepcopy(b) for p in sub_populations for b in p]
def _evaluatePopulationWorker(population, desired_result, fitness, loss_function):
	print("_evaluatePopulationWorker - ", str(mp.current_process()))
	for b in population:
		if _shapeIsIllegal(b['verts']): 
			b['fitness'] = float('Inf')
			b['nrpx'] = 0
			continue
		b['fitness'], b['nrpx'] = _getEfFitness(b['verts'], desired_result, fitness, mode = loss_function)
	return population
def _splitWorkload(population, nr_workers):
	'''source: http://wordaligned.org/articles/slicing-a-list-evenly-with-python'''
	l = len(population)
	assert 0 < nr_workers <= l
	workload, remainder = divmod(l, nr_workers)
	sub_populations = [population[p:p+workload] for p in range(0, l, workload)]
	sub_populations[nr_workers-1:] = [population[-remainder-workload:]]
	return copy.deepcopy(sub_populations) # is this necessary?
'''
_sectorMeanCrossover
separates the polar coordinates of the parents into child_nr_verts sectors
and averages the coordinates in each vector. If a vector does not contain any
vertices, then no vertex is added to the child vertices in that sector.
'''
def _sectorMeanCrossover(p1, p2, Cx, Cy):
	# cross over within specific section of unit circle
	child = {}
	child_p_verts = []
	v1_p = np.array(convert2Polar(p1['verts'], Cx, Cy))
	v2_p = np.array(convert2Polar(p2['verts'], Cx, Cy))
	# child has nr vertices in range between both parents
	parent_lengths = sorted([len(v1_p), len(v2_p)])
	child_nr_verts = random.randint(parent_lengths[0], parent_lengths[1])
	# for each child vertex we define a sector (sectors are evenly spaced)
	sector_size = 2*np.pi/child_nr_verts
	sector_offset = random.random()*sector_size*0.99 # ensure that offset doesn't exceeds 2pi
	v_combined = np.concatenate((v1_p, v2_p))
	for sector in range(child_nr_verts):
		sector_start = sector*sector_size + sector_offset
		sector_end = (sector + 1)*sector_size + sector_offset
		in_sector = _inRangePolarVerts(v_combined, sector_start, sector_end)
		# average vertices in sector and add new vertex to child['verts']
		#from IPython import embed; embed()
		if sum(in_sector) > 0:	
			child_p_verts.append(tuple(np.average(v_combined[in_sector], axis=0)))
	child_p_verts = sorted(child_p_verts,key=lambda l:l[1])
	child['verts'] = convertFromPolar(child_p_verts, img_center, img_center)
	child['age'] = max(p1['age'], p2['age'])
	return child
def _inRangePolarVerts(v_p, sector_start, sector_end):
	assert sector_start < sector_end
	assert sector_start < 2*np.pi
	if 2*np.pi > sector_end:
		in_sector = np.logical_and(v_p[:, 1] > sector_start, v_p[:,1] < sector_end)
	else:
		sector_part_1 = np.logical_and(v_p[:, 1] > sector_start, v_p[:,1] < 2*np.pi)
		sector_part_2 = np.logical_and(v_p[:, 1] > 0, v_p[:,1] < sector_end%(2*np.pi))
		in_sector = np.logical_or(sector_part_1, sector_part_2)
	return in_sector

def _sliceCrossover(p1, p2, Cx, Cy):
	child = {}
	v1_p = convert2Polar(p1['verts'], Cx, Cy)
	v2_p = convert2Polar(p2['verts'], Cx, Cy)

	p2_orig_size = len(v2_p)
	p1_orig_size = len(v1_p)

	start_idx = random.randint(0,p1_orig_size-1)	
	end_idx = random.randint(0,p1_orig_size-1)
	timeout_counter = 10
	while end_idx==start_idx and timeout_counter > 0:
		end_idx = random.randint(0,p1_orig_size-1)
		timeout_counter -= 1

	if timeout_counter < 1:
		print("_sliceCrossover(): timeout_counter exceeded", p1)

	min_theta = v1_p[start_idx][1]
	max_theta = v1_p[end_idx][1]

	# ensure wrap around for indecies
	if start_idx < end_idx:
		copy_range = range(start_idx, end_idx) 
	else:
		copy_range = range(start_idx, end_idx+p1_orig_size)	
	#ensure wrap around angle range
	if min_theta < max_theta:
		v2_p = list(filter(lambda a: a[1] < min_theta or a[1] > max_theta, v2_p))
	else:
		v2_p = list(filter(lambda a: a[1] < min_theta and a[1] > max_theta, v2_p))
	for i in copy_range:
		if len(v2_p) < maxNumVerts:
			v2_p.append(v1_p[i%p1_orig_size])
		else:
			break
	v2_p = sorted(v2_p,key=lambda l:l[1])
	child['verts'] = convertFromPolar(v2_p, Cx, Cy)
	child['age'] = max(p1['age'], p2['age'])
	return child
def _getSimilarityOfShape(p1, p2):
	eta = 1e-10
	im1, nrpx1 = getimg(p1['verts'], color='bw', fill=True, img_res = 128)
	im2, nrpx2 = getimg(p2['verts'], color='bw', fill=True, img_res = 128)
	np_im1 = np.array(im1)
	np_im2 = np.array(im2)
	diff = sum(np.absolute(np_im1-np_im2).flatten()) 
	area = np.maximum(np.mean((nrpx1, nrpx2)), eta) 
	return diff/area
'''
_ShapeIsIllegal(verts): returns true if
	- shape has less than 3 vertices
	- is out of bounds of the search space
	- is self-intersecting
'''
def _shapeIsIllegal(verts):
	if len(verts) < 3:
		return True
	v_polar = convert2Polar(verts, img_center, img_center)
	for i in range(len(v_polar)):
		if v_polar[i][0] < minRad or v_polar[i][0] > maxRad:
			return True
	if not isNonIntersecting(verts):
		return True
def _smoothenSpikeyness(verts):
	p_verts = np.array(convert2Polar(verts, img_center, img_center))
	avg_rad = np.mean(p_verts, axis=0)[0]
	p_verts[:,0] = (p_verts[:,0]+avg_rad)/2
	return convertFromPolar(p_verts, img_center, img_center)	
def _getSpikeyness(verts):
	p_verts = np.array(convert2Polar(verts, img_center, img_center))
	std_rad = np.std(p_verts, axis=0)[0]
	return std_rad
def isExceedingNeighbour(verts, idx, cur_theta):
	isExceedingPrevious = False
	isExceedingNext = False
	if verts[idx][1] > verts[idx-1][1]:
		isExceedingPrevious = cur_theta  <= verts[idx-1][1]
	else:
		isExceedingPrevious = (cur_theta + 2*np.pi) <= verts[idx-1][1]

	next_idx = (idx+1) % len(verts)  
	if verts[idx][1] < verts[next_idx][1]:
		isExceedingNext = cur_theta  >= verts[next_idx][1]
	else:
		isExceedingNext = cur_theta >= (verts[next_idx][1] + 2*np.pi)
	return isExceedingPrevious or isExceedingNext
def convert2Polar(verts, center_X = 0, center_Y = 0 ):
	np_verts = np.array(verts)
	# center vertices by subtracting the center point from each vertex
	np_center_x = np.ones(np_verts.shape[0])*center_X
	np_center_y = np.ones(np_verts.shape[0])*center_Y
	np_center = np.stack((np_center_x, np_center_y), axis = 1)
	centered_verts = np_verts - np_center
	np_rad = np.sqrt(np.sum(np.power(centered_verts,2), axis=1))
	np_theta = np.arctan2(centered_verts[:,1], centered_verts[:,0])
	np_theta = convert2positiveRad(np_theta)
	verts = [(rad, theta) for rad, theta in zip(np_rad, np_theta)]
	return verts
def convert2positiveRad(np_theta):
	return np.mod((np_theta+2*np.pi),2*np.pi)
def convertFromPolar(verts, center_X = 0, center_Y = 0 ):
	np_verts = np.array(verts)
	np_x = center_X + np.round(np_verts[:,0]*np.cos(np_verts[:,1]))
	np_y = center_Y + np.round(np_verts[:,0]*np.sin(np_verts[:,1]))
	verts = [(int(x), int(y)) for x,y in zip(np_x, np_y)]
	return verts
def _getEfFitness(verts, desired_result, fitness, mode = 'MAPE'):
	img, nrpx = getimg(verts, color='grayscale', fill=True, img_res = 128)
	res = torch.squeeze(fitness.evaluate(img))[0:len(desired_result)]
	des = torch.tensor(desired_result)
	#max_area = np.pi*np.power(maxRad, 2)
	#area_error = np.maximum(nrpx-max_area, 0)/max_area
	if mode == 'MAPE':
		error = torch.abs(res-des)/des 
		sample_fitness = torch.mean(error).item()
	elif mode == 'MAE':
		error = torch.abs(res-des) 
		sample_fitness = torch.mean(error).item()
	elif mode == 'MSE':
		error = torch.pow(res-des, 2)
		sample_fitness = torch.mean(error).item()
	elif mode == 'ED':
		error = torch.dist(res.float(),des.float()) 
		sample_fitness = torch.mean(error).item()
	return sample_fitness, nrpx
def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ) :
	'''
	generatePolygon() and clip() were adapted from this StackOverflow answer
	source: https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
	Start with the centre of the polygon at ctrX, ctrY, 
	then creates the polygon by sampling points on a circle around the centre. 
	Randon noise is added by varying the angular spacing between sequential points,
	and by varying the radial distance of each point from the centre.

	Params:
	ctrX, ctrY - coordinates of the "centre" of the polygon
	aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
	irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
	spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
	numVerts - self-explanatory

	Returns a list of vertices, in CCW order.
	'''

	irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
	spikeyness = clip( spikeyness, 0,1 ) * aveRadius

	# generate n angle steps
	angleSteps = []
	lower = (2*math.pi / numVerts) - irregularity
	upper = (2*math.pi / numVerts) + irregularity
	sum = 0
	for i in range(numVerts) :
		tmp = random.uniform(lower, upper)
		angleSteps.append( tmp )
		sum = sum + tmp

	# normalize the steps so that point 0 and point n+1 are the same
	k = sum / (2*math.pi)
	for i in range(numVerts) :
		angleSteps[i] = angleSteps[i] / k

	# now generate the points
	points = []
	angle = random.uniform(0, 2*math.pi)
	for i in range(numVerts) :
		r_i = clip( random.gauss(aveRadius, spikeyness), 0, aveRadius )
		if (r_i > 64):
			print("Unexpected State: max value should not be exceeded", r_i)
		x = ctrX + r_i*math.cos(angle)
		y = ctrY + r_i*math.sin(angle)
		points.append( (int(x),int(y)) )

		angle = angle + angleSteps[i]

	return points

def clip(x, min, max) :
	if( min > max ) :  return x    
	elif( x < min ) :  return min
	elif( x > max ) :  return max
	else :             return x

def getimg(verts, color, fill, img_res = 128, raw = False):
	if (color == 'grayscale'):
		#c1 = (100,100,100) OLD VALUES DON"T WORK ON NEWLY TRAINED NETWORKS
		#c2=(156,156,156)
		c1 = (25,25,25)
		c2=(230,230,230)
	else:
		c1 = (0,0,0)
		c2=(255,255,255)
	im = Image.new('RGB', (img_res, img_res), c2)
	draw = ImageDraw.Draw(im)
	
	# either use .polygon(), if you want to fill the area with a solid colour
	if (fill):
		draw.polygon( verts, outline=c1,fill=c1 )
	else:
		draw.polygon( verts, outline=c1,fill=c2 )

	# Count Nr of pixels made up by cross-section
	nrpx = im.histogram()[c1[0]]

	if raw:
		return im, nrpx
	#im.show()
	#im = im.convert('LA')
	im = io._plugins.pil_plugin.pil_to_ndarray(im)
	im = rgb2gray(im)
	#Image.fromarray(im).show()
	image = im.transpose((0, 1))
	im = torch.from_numpy(image).float()
	im = torch.unsqueeze(im,0)
	im = torch.unsqueeze(im,1)
	#img_dim = image.shape
	# or .line() if you want to control the line thickness, or use both methods together!
	#draw.line( tupVerts+[tupVerts[0]], width=2, fill=black )
	return(im, nrpx)
'''
Functions to check for self intersection of polygons
'''
def ccw(a, b, c):
	return( (b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1]) )

def intersects(a,b,c,d):
	if (ccw(a, b, c)*ccw(a,b,d) >= 0):
		return False
	elif (ccw(c, d, a)*ccw(c,d,b) >= 0):
		return False
	else:
		return True

def isNonIntersecting(verts):
	for id1 in range(1, len(verts)+1):
		for id2 in range(1, len(verts)+1):
			if intersects(verts[id1-1],verts[id1%len(verts)], verts[id2-1], verts[id2%len(verts)]):
				return False
	return True
'''
Function to calculate are of triangle described by vertices a, b, c
'''
def triangleArea(a, b, c):
	return abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))/2
# remove vertices from list if they are 'almost' colinear
def removeQuasiColinearVertices(v, eta = 1e-5):
	i = 0
	while ( i < len(v) ):
		if  len(v) > 3 and triangleArea(v[i], v[i-1], v[i-2]) < eta:
			v.remove(v[i-1])
			i = 0
			continue
		i += 1
	return v
'''
Function to evaluate angle between any 3 coordinates
'''
def dist(v1, v2):
	return np.linalg.norm(v1-v2)
def get_angle_between_coordinates(v1, v2, v3):
	#get angle by law of cosines
	law_cosines = (dist(v1,v2)**2 + dist(v2,v3)**2 - dist(v1,v3)**2) / (2 * dist(v1,v2) * dist(v2,v3))
	if law_cosines > 1 or law_cosines < -1:
		#print("Line 799: OOBError from law_cosines {}:\n v1 {} v2 {} v3 {} with dist(v1,v2) {} dist (v2,v3) {} dist(v1,v3) {}".format(
		#		law_cosines, v1, v2, v3, dist(v1,v2), dist(v2,v3), dist(v1,v3)))
		return np.pi
	return np.arccos(law_cosines)
def removeAcuteAngle(verts, theta_threshold = np.radians(15)):
	v = np.asarray(verts)
	i = 0
	while ( i < len(v) ):
		if  len(v) > 2:
			theta = get_angle_between_coordinates(v[i], v[i-1], v[i-2])
			if theta < theta_threshold or theta > (2*np.pi - theta_threshold):
				v = np.delete(v,i-1, 0)
				i = 0
				continue
		i += 1
		# return array of tuples
	return [(e[0], e[1]) for e in v]
'''
Removes duplicated vertices
'''
def removeDuplicates(v, respect_order = True):
	v = list(OrderedDict.fromkeys(v)) 
	v_polar = convert2Polar(v, img_center, img_center)
	# sort vertices by angle to avoid self-overlapping shapes:
	v_polar = sorted(v_polar, key = lambda v: v[1])
	return convertFromPolar(v_polar, img_center, img_center)

def _writePlotFile(best_verts_history = None, str_option = ""):
	if best_verts_history is not None:
		plot_file = plot_file_folder / (
					str(mp.current_process()).strip('<SpawnProcess(').strip(', started daemon)>') + str_option + plot_file_ext	)
		with open(plot_file, mode='w') as plt_file:
			plt_writer = csv.writer(plt_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			plt_writer.writerows(best_verts_history)

def _writeToOptimizationLog(epoch, performance, desired_result, time_string = None, optional_str = None,  ef = None):
	# result_path
	#ToDo: Implement logging function to create consistent optimization logs
	if time_string is None:
		time_string = time.strftime('%Y%m%d_%H-%M', time.localtime(time.time()))
	if optional_str is not None:
		time_string = time_string + '_' + optional_str
	file_name = "OptimLog_{}_{}.csv".format(str(desired_result).strip('[]').replace(',','-'), time_string)
	log_file = result_path / file_name
	with open(log_file, mode='a') as my_log:
		plt_writer = csv.writer(my_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		plt_writer.writerow([epoch, performance, desired_result])
def _WriteEvolutionLog(epoch, population, desired_result = "", time_string = ""):
	file_name = "EvolutionLog_{}_{}_{}.age_pareto".format(
		len(population),
		str(desired_result).strip('[]').replace(',','-'),
		time_string)
	evo_log_file = result_path / "Evolution" / file_name
	file_exists = True if evo_log_file.exists() else False
	with open(evo_log_file, mode='a') as evolution_log:
		writer = csv.DictWriter( evolution_log,
			fieldnames=["epoch", "age", "fitness"])
		if not file_exists:	
			writer.writeheader()
		for p in population:
			writer.writerow({"epoch": epoch,
							 "age":p['age'],
							 "fitness":p['fitness']})
def write_datapoint(i, verts, mape_err, ef123, desired_result, result_dataset_root):
	# writes found vertices to file
	cur_datapt_path = result_dataset_root / str(i)
	Path(cur_datapt_path / "img").mkdir(parents=True, exist_ok=True)
	Path(cur_datapt_path / "numbers").mkdir(parents=True, exist_ok=True) 
	Path(cur_datapt_path / "label").mkdir(parents=True, exist_ok=True)
	# write the vertices into the datapoint folder (center them around 0)
	np.save(Path(cur_datapt_path / 'numbers' / 'verts.npy'),
				np.array([[v[0]-img_center, v[1]-img_center] for v in verts]))
	# store MAPE attained for solution during optimization
	np.save(Path(cur_datapt_path / 'numbers' / 'mape_optim.npy'),
				np.array(mape_err))
	# find and store desired EF123
	np.save(Path(cur_datapt_path / 'numbers' / 'ef123_desired.npy'),
				np.array(desired_result))
	# find and store predicted EF123
	np.save(Path(cur_datapt_path / 'numbers' / 'ef123_predicted.npy'),
				ef123.numpy())

def main(model_name = None, GPU = 0, neuralnetwork = 'ConvNet', mode = 'R', nr_runs = 1, desired_results_file = None):
	max_iter = 2*50000#8*25000
	GPUs = [0,1,2]
	#GPUs = [0,1,2,3,4,5,6,7]
	workers_per_gpu = 3
	num_classes = 3
	loss_function = 'MAE'

	#Only write a plot file if this is a single run
	global write_plot_file
	write_plot_file = (nr_runs == 1)

	if desired_results_file is None:
		desired_result = [78.00118062,109.5854501,482.0235725] # Exp02_Test1 (Arbitrary Datapoint)
		result_file = result_path / 'optimization_result_{}_{}_{}.csv'.format(str(desired_result).strip('[]').replace(',','-'),mode, time_string)
		result_dataset_root = result_path / 'optimization_result_{}_{}_{}'.format(str(desired_result).strip('[]').replace(',','-'),mode, time_string)
		#desired_result = [123.9730781,146.0302206,750.0447463] # Exp02_Test2 (Arbitrary Datapoint)
		#desired_result = [204.8386207,212.5732456,1170.448762] # Exp02_Test3 (Arbitrary Datapoint)
		# define optimization job parameters
		job_params = (mode, desired_result, max_iter, loss_function, maxNumVerts, ALLOW_VERT_REMOVAL)	
	else:
		with open(desired_results_file, mode='r') as json_file:
			desired_results_dict = json.load(json_file)
		job_params = [(mode, desired_result, max_iter, loss_function, maxNumVerts, ALLOW_VERT_REMOVAL) for desired_result in desired_results_dict.values()]
		desc_str = 'MULTIRESULT_{}'.format(len(job_params))
		result_file = result_path / 'optimization_result_{}_{}_{}.csv'.format(desc_str,mode, time_string)
		result_dataset_root = result_path / 'optimization_result_{}_{}_{}'.format(desc_str,mode, time_string)

	# define GPU evaluator instance configuration
	eval_config = {'GPUs': GPUs}
	eval_config['workers_per_gpu'] = workers_per_gpu
	eval_config['model_name'] = model_name
	eval_config['NN'] = neuralnetwork
	eval_config['num_classes'] = num_classes
	eval_config['img_res'] = img_res

	# find results
	results = findConfiguration(job_params, eval_config = eval_config, num_results = nr_runs)

	fitness = NNEvaluator(eval_config['model_name'], eval_config['GPUs'][0], eval_config['NN'], 
								num_classes=eval_config['num_classes'], img_res = eval_config['img_res'])
	with open(result_file, mode='w') as res_file:
		plt_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		if type(results) is tuple:
			r = results	
			img, nrpx = getimg(r[0], color='grayscale', fill=True, img_res = img_res)
			res = torch.squeeze(fitness.evaluate(img))[0:len(r[2])]
			plt_writer.writerow([r[0],r[1], res])
		elif type(results) is list:
			for i, r in enumerate(results):
				print("result from worker", i, "fitness:",r[1], "isNonIntersecting:", isNonIntersecting(r[0]))
				img, nrpx = getimg(r[0], color='grayscale', fill=True, img_res = img_res)
				res = torch.squeeze(fitness.evaluate(img))[0:len(r[2])]
				plt_writer.writerow([r[0],r[1], res])
				write_datapoint(i, r[0], r[1], res, r[2], result_dataset_root)

	del fitness
	print("main() completed")
	return 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='BeamOptimizer Parser')
	parser.add_argument('--model-name', type=str, default=None, metavar='model.ckpt',
					  help='Provide a .ckpt model file.')
	parser.add_argument('--GPU', type=int, default=0, metavar='Any valid GPU id e.g. 0',
					  help='Provide a GPU id e.g. 0,1,etc.')
	parser.add_argument('--NN', type=str, default='ConvNet', metavar='name of the file that contains the ConvNet class',
					  help='Provide the name of the file containing the ConvNet class without file-extension')
	parser.add_argument('--mode', type=str, default='R', metavar='which optimization mode should be used?',
					  help='R for Random, H for hillClimber')
	parser.add_argument('--runs', type=int, default='1', metavar='# of runs to perform',
					  help='# runs to perform e.g. enter 10 for a ten runs')
	parser.add_argument('--desired-results', type=str, default=None, metavar='desired_results.json',
					  help='file containing a dictionary of {<beam_idx>: [ef1, ef2, ef3] ...}')
	args = parser.parse_args()
	start_time = time.time()
	main(args.model_name, args.GPU, args.NN, args.mode, args.runs, args.desired_results)
	print("if __name__ == '__main__': COMPLETED in {} seconds".format(time.time()-start_time))

'''
DEBUG CODE:
Loads existing sample and can be used to feed in right solution to test code
'''
# sample dictionary of existing datapoint
def load_existing_verts(dp_nr = None):
	DATASET_PATH = Path("/media/ron/DataSSD/datasets/TwistedBeamDS/{}".format(dp_nr))
	# print eigenfrequencies of loaded point
	ef1 = np.load(DATASET_PATH / 'label' / 'ef1.npy')
	ef2 = np.load(DATASET_PATH / 'label' / 'ef2.npy')
	ef3 = np.load(DATASET_PATH / 'label' / 'ef3.npy')
	ef123 = [ef1[0], ef2[0], ef3[0]]
	print("Data point {} has EF123:{}".format(dp_nr, ef123))
	verts = np.load(DATASET_PATH / 'numbers' / 'verts.npy')
	return [(int(v[0]+img_center), int(v[1]+img_center)) for v in verts]
'''
TRASH BIN
'''
'''
def getVertsOOBError(verts, img_res = 128, normalize = True):
	error = 0
	np_verts = np.array(verts)
	center_point = np.ones(np_verts.shape)*img_center
	np_verts = np_verts-center_point
	np_rads = np.sqrt(np.sum(np.power(np_verts,2),1))
	overmax = np_rads - np.ones(np_rads.shape)*maxRad
	undermin = np.ones(np_rads.shape)*minRad - np_rads 
	if normalize:
		overmax = overmax / maxRad
		undermin = undermin / minRad
	np_error = np.stack((overmax, undermin, np.zeros(overmax.shape)), axis = 1)
	error = np.sum(np.amax(np_error, axis = 1))
	return error
'''
