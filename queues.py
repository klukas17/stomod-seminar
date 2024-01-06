import heapq
import numpy as np


class Generator:
    def __init__(self, generator_function):
        self.generator_function = generator_function


class Consumer:
    def __init__(self, consumer_function):
        self.consumer_function = consumer_function


class Processor:
    def __init__(self, number_of_processors):
        self.number_of_processors = number_of_processors
        self.number_of_jobs = 0
        self.jobs = set()


class Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.number_of_jobs = 0
        self.jobs = []


class Queue:
    def __init__(self, queue_id, generator_function, consumer_function, number_of_processors, buffer_size):
        self.queue_id = queue_id
        self.generator = Generator(generator_function) if generator_function is not None else None
        self.consumer = Consumer(consumer_function) if consumer_function is not None else None
        self.processor = Processor(number_of_processors)
        self.buffer = Buffer(buffer_size)


class Logger:
    def __init__(self):
        self.logs = []

    def add_log(self, log):
        self.logs.append(log)

    def calculate_average_number_of_busy_processors(self, queue_id, end_time):
        sum = 0
        last_time = 0
        num = 0

        for log in self.logs:
            if log.queue_id == queue_id and log.event_description in ['ENTER_PROCESSOR', 'EXIT_PROCESSOR']:
                time = log.time - last_time
                last_time = log.time
                sum += time * num
                if log.event_description == 'ENTER_PROCESSOR':
                    num += 1
                else:
                    num -= 1

        return sum / end_time

    def calculate_stationary_distribution(self, queue_id, end_time, number):
        sum = 0
        last_time = 0
        num = 0

        for log in self.logs:
            if log.queue_id == queue_id and log.event_description in ['ENTER_QUEUE', 'EXIT_PROCESSOR', 'EXIT_SYSTEM_FORCED']:
                time = log.time - last_time
                last_time = log.time
                sum += time * (num == number)
                if log.event_description == 'ENTER_QUEUE':
                    num += 1
                else:
                    num -= 1

        return sum / end_time

    def calculate_stationary_distribution_global(self, end_time, number):
        sum = 0
        last_time = 0
        num = 0

        for log in self.logs:
            if log.event_description in ['ENTER_SYSTEM', 'EXIT_SYSTEM', 'EXIT_SYSTEM_FORCED']:
                time = log.time - last_time
                last_time = log.time
                sum += time * (num == number)
                if log.event_description == 'ENTER_SYSTEM':
                    num += 1
                else:
                    num -= 1

        return sum / end_time

    def calculate_probability_of_loosing_a_job(self, queue_id):
        jobs_entered = 0
        jobs_lost = 0

        for log in self.logs:
            if log.queue_id == queue_id and log.event_description == 'ENTER_QUEUE':
                jobs_entered += 1
            elif log.queue_id == queue_id and log.event_description == 'EXIT_SYSTEM_FORCED':
                jobs_lost += 1

        return jobs_lost / jobs_entered

    def calculate_probability_of_loosing_a_job_global(self):
        jobs_entered = 0
        jobs_lost = 0

        for log in self.logs:
            if log.event_description == 'ENTER_SYSTEM':
                jobs_entered += 1
            elif log.event_description == 'EXIT_SYSTEM_FORCED':
                jobs_lost += 1

        return jobs_lost / jobs_entered


    def calculate_probability_of_a_job_waiting(self, queue_id, end_time, s):
        sum = 0
        last_time = 0
        num = 0

        for log in self.logs:
            if log.queue_id == queue_id and log.event_description in ['ENTER_PROCESSOR', 'EXIT_PROCESSOR']:
                time = log.time - last_time
                last_time = log.time
                sum += time * (num == s)
                if log.event_description == 'ENTER_PROCESSOR':
                    num += 1
                else:
                    num -= 1

        return sum / end_time

    def calculate_expected_number_of_jobs_in_buffer(self, queue_id, end_time):
        sum = 0
        last_time = 0
        num = 0

        for log in self.logs:
            if log.queue_id == queue_id and log.event_description in ['ENTER_BUFFER', 'EXIT_BUFFER']:
                time = log.time - last_time
                last_time = log.time
                sum += time * num
                if log.event_description == 'ENTER_BUFFER':
                    num += 1
                else:
                    num -= 1

        return sum / end_time

    def calculate_expected_waiting_time_in_buffer(self, queue_id):
        m = {}

        for log in self.logs:
            if log.queue_id == queue_id:
                if log.job_id not in m:
                    m[log.job_id] = [None, None, None, None]
                match log.event_description:
                    case 'ENTER_BUFFER':
                        m[log.job_id][0] = log.time
                    case 'EXIT_BUFFER':
                        m[log.job_id][1] = log.time
                    case 'ENTER_PROCESSOR':
                        m[log.job_id][2] = log.time
                    case 'EXIT_PROCESSOR':
                        m[log.job_id][3] = log.time
                    case _:
                        pass

        sum = 0

        for k in m:
            if m[k][0] is not None and m[k][1] is not None:
                sum += m[k][1] - m[k][0]

        return sum / len(m)


class Event:
    def __init__(self, job_id, queue_id, event_description, time):
        self.job_id = job_id
        self.queue_id = queue_id
        self.event_description = event_description
        self.time = time

    def __lt__(self, other):
        return self.time < other.time

    def __le__(self, other):
        return self.time <= other.time

    def __eq__(self, other):
        return self.time == other.time

    def __ne__(self, other):
        return self.time != other.time

    def __gt__(self, other):
        return self.time > other.time

    def __ge__(self, other):
        return self.time >= other.time

    def __str__(self):
        return f"Event: job_id={self.job_id}, queue_id={self.queue_id}, event_description={self.event_description}, time={self.time}"


class Job:
    def __init__(self, job_id):
        self.job_id = job_id


class Exit:
    def __init__(self):
        self.queue_id = -1


class QueueNetwork:
    def __init__(self):
        self.exit = Exit()
        self.logger = Logger()
        self.queue_counter = 0
        self.job_counter = 0
        self.queues = {}
        self.transitions = {}

    def create_queue(self, generator_function, consumer_function, number_of_processors, buffer_size):
        queue = Queue(self.queue_counter, generator_function, consumer_function, number_of_processors, buffer_size)
        self.queues[self.queue_counter] = queue
        self.queue_counter += 1
        return queue

    def create_job(self):
        job = Job(self.job_counter)
        self.job_counter += 1
        return job

    def set_transition(self, queue, transition):
        self.transitions[queue.queue_id] = transition

    def simulate(self, end_time):
        time = 0
        event_queue = []
        heapq.heapify(event_queue)

        for i in self.queues:
            queue = self.queues[i]
            if queue.generator is not None:
                job = self.create_job()
                event = Event(job.job_id, queue.queue_id, 'ENTER_QUEUE', queue.generator.generator_function())
                if event.time < end_time:
                    heapq.heappush(event_queue, Event(job.job_id, queue.queue_id, 'ENTER_SYSTEM', event.time))
                    heapq.heappush(event_queue, event)

        while len(event_queue) > 0:
            event = heapq.heappop(event_queue)
            self.logger.add_log(event)
            time = event.time
            queue = self.queues[event.queue_id]

            match event.event_description:
                case 'ENTER_QUEUE':
                    if queue.generator is not None:
                        job = self.create_job()
                        new_event = Event(job.job_id, event.queue_id, 'ENTER_QUEUE', time + queue.generator.generator_function())
                        if new_event.time < end_time:
                            heapq.heappush(event_queue, Event(job.job_id, queue.queue_id, 'ENTER_SYSTEM', new_event.time))
                            heapq.heappush(event_queue, new_event)

                    if queue.processor.number_of_jobs < queue.processor.number_of_processors:
                        new_event = Event(event.job_id, event.queue_id, 'ENTER_PROCESSOR', time)
                        if event.time < end_time:
                            queue.processor.number_of_jobs += 1
                            heapq.heappush(event_queue, new_event)

                    elif queue.buffer.number_of_jobs < queue.buffer.capacity:
                        new_event = Event(event.job_id, event.queue_id, 'ENTER_BUFFER', event.time)
                        queue.buffer.number_of_jobs += 1
                        heapq.heappush(event_queue, new_event)

                    else:
                        heapq.heappush(event_queue, Event(event.job_id, event.queue_id, 'EXIT_SYSTEM_FORCED', event.time))

                case 'ENTER_PROCESSOR':
                    new_event = Event(event.job_id, event.queue_id, 'EXIT_PROCESSOR', time + queue.consumer.consumer_function())
                    if new_event.time < end_time:
                        queue.processor.jobs.add(event.job_id)
                        heapq.heappush(event_queue, new_event)

                case 'EXIT_PROCESSOR':
                    if queue.buffer.number_of_jobs == 0:
                        queue.processor.number_of_jobs -= 1
                    else:
                        heapq.heappush(event_queue, Event(event.job_id, event.queue_id, 'RELEASE_FROM_BUFFER', event.time))
                    queue.processor.jobs.remove(event.job_id)

                    transition_probabilities = self.transitions[queue.queue_id]
                    choices = list(transition_probabilities.keys())
                    weights = list(transition_probabilities.values())
                    next_queue_id = np.random.choice(choices, p=weights)
                    if next_queue_id == -1:
                        heapq.heappush(event_queue, Event(event.job_id, event.queue_id, 'EXIT_SYSTEM', event.time))
                    else:
                        heapq.heappush(event_queue, Event(event.job_id, next_queue_id, 'ENTER_QUEUE', event.time))

                case 'ENTER_BUFFER':
                    queue.buffer.jobs.append(event.job_id)

                case 'EXIT_BUFFER':
                    new_event = Event(event.job_id, event.queue_id, 'ENTER_PROCESSOR', time)
                    if event.time < end_time:
                        heapq.heappush(event_queue, new_event)

                case 'RELEASE_FROM_BUFFER':
                    queue.buffer.number_of_jobs -= 1
                    job_id = queue.buffer.jobs[0]
                    queue.buffer.jobs = queue.buffer.jobs[1:]
                    heapq.heappush(event_queue, Event(job_id, event.queue_id, 'EXIT_BUFFER', event.time))

                case 'ENTER_SYSTEM':
                    pass

                case 'EXIT_SYSTEM':
                    pass

                case 'EXIT_SYSTEM_FORCED':
                    pass

                case _:
                    pass
