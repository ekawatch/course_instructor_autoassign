import pandas as pd
import numpy as np
from tqdm import tqdm
import numba
from joblib import Parallel, delayed
import sys

# ตรวจสอบว่ามี GPU ที่ Numba (CUDA) สามารถใช้งานได้หรือไม่
HAS_GPU = False
try:
    # Check if numba.cuda exists and is available
    if hasattr(numba, 'cuda'):
        if numba.cuda.is_available():
            HAS_GPU = True
            print("ตรวจพบ GPU ที่รองรับ CUDA, จะใช้ GPU ในการคำนวณ Fitness")
        else:
            print("Numba พบโมดูล CUDA แต่ไม่สามารถใช้งานได้ (เช่น ไม่มี GPU หรือไดรเวอร์มีปัญหา)")
            print("จะใช้ CPU ในการคำนวณ Fitness แบบขนาน")
except numba.cuda.CudaSupportError as e:
    print(f"Numba CUDA Support Error: {e}")
    print("ไม่พบ GPU ที่รองรับ CUDA หรือการตั้งค่าไม่ถูกต้อง, จะใช้ CPU ในการคำนวณ Fitness แบบขนาน")
except Exception as e:
    # Catch any other exceptions during the check
    print(f"เกิดข้อผิดพลาดในการตรวจสอบ CUDA: {e}")
    print("ไม่พบ GPU ที่รองรับ CUDA, จะใช้ CPU ในการคำนวณ Fitness แบบขนาน")


# ============== 1. สร้างข้อมูลตัวอย่าง ==============
def create_sample_data(num_courses=60, num_instructors=20):
    """สร้างข้อมูลตัวอย่างสำหรับรายวิชาและอาจารย์"""
    # Set a random seed for reproducibility
    np.random.seed(42)

    # --- ข้อมูลวิชา ---
    course_ids = [f"SCMA{100+i}" for i in range(num_courses)]
    course_credits = np.random.randint(1, 5, size=num_courses)
    years = np.random.randint(1, 5, size=num_courses)
    semesters = np.random.randint(1, 3, size=num_courses)
    # 0=General, 1=Compulsory, 2=Elective
    course_types = np.random.choice([0, 1, 2], size=num_courses, p=[0.2, 0.5, 0.3])

    courses_df = pd.DataFrame({
        'courseID': course_ids,
        'courseCredit': course_credits,
        'year': years,
        'semester': semesters,
        'courseType': course_types
    })

    # --- ข้อมูลอาจารย์ ---
    instructor_ids = list(range(num_instructors))
    instructor_names = [f"Instructor_{i}" for i in instructor_ids]

    # สุ่ม personal_credits ก่อน
    personal_credits = np.random.randint(6, 13, size=num_instructors).astype(float) # ใช้ float เพื่อให้ปรับค่าได้แม่นยำ

    # คำนวณผลรวม credit ของทุกวิชา
    total_course_credits = courses_df['courseCredit'].sum()

    # คำนวณผลรวม personal_credits ที่สุ่มได้
    total_personal_credits = personal_credits.sum()

    # คำนวณผลต่างและค่าเฉลี่ยที่จะใช้ปรับ
    credit_difference = total_course_credits - total_personal_credits
    average_adjustment = credit_difference / num_instructors

    # ปรับ personal_credits ของแต่ละอาจารย์
    personal_credits += average_adjustment

    # ตรวจสอบให้แน่ใจว่า personal_credits ไม่ติดลบ ( unlikely with current ranges but good practice)
    personal_credits[personal_credits < 0] = 0

    instructors_df = pd.DataFrame({
        'InstructorID': instructor_ids,
        'name': instructor_names,
        'personalCredit': personal_credits
    })

    # --- สร้าง Preference Score ---
    preference_matrix = np.zeros((num_instructors, num_courses), dtype=int)
    for i in range(num_instructors):
        # อาจารย์แต่ละคนมี preference ไม่เกิน 10 วิชา
        num_prefs = np.random.randint(50, 80)
        pref_courses_indices = np.random.choice(range(num_courses), num_prefs, replace=False)

        # กำหนดค่า 1 (อยากสอน) และ -1 (ไม่อยากสอน)
        num_wants = np.random.randint(1, num_prefs)
        want_indices = np.random.choice(pref_courses_indices, num_wants, replace=False)
        dont_want_indices = np.setdiff1d(pref_courses_indices, want_indices)

        preference_matrix[i, want_indices] = 1
        preference_matrix[i, dont_want_indices] = -1

    pref_df = pd.DataFrame(preference_matrix, columns=courses_df['courseID'])
    instructors_df = pd.concat([instructors_df, pref_df, ], axis=1)

    return courses_df, instructors_df

# ============== 2. Fitness Function (ออกแบบสำหรับ Numba) ==============
# ฟังก์ชันนี้ถูกออกแบบให้ทำงานกับ NumPy arrays เพื่อประสิทธิภาพสูงสุด
# และสามารถคอมไพล์ด้วย Numba JIT ได้
@numba.jit(nopython=True)
def calculate_fitness_for_chromosome(
    chromosome,
    course_credits,
    course_year_semester_type, # รวม year, semester, courseType
    instructor_personal_credits,
    preference_matrix):
    """
    คำนวณค่า Fitness สำหรับ Chromosome หนึ่งตัว
    """
    num_instructors = len(instructor_personal_credits)
    num_courses = len(course_credits)

    # --- FN1: ผลต่างหน่วยกิต ---
    assigned_credits_per_instructor = np.zeros(num_instructors)
    for i in range(num_courses):
        instructor_id = chromosome[i]
        assigned_credits_per_instructor[instructor_id] += course_credits[i]

    fn1 = np.sum((assigned_credits_per_instructor - instructor_personal_credits) ** 2)

    # --- FN2: วิชาบังคับซ้อนในเทอมและชั้นปีเดียวกัน ---
    fn2 = 0
    # ใช้ tuple array ที่ซับซ้อนไม่ได้ใน nopython mode, จึงต้องวนลูปตรวจสอบ
    for inst_id in range(num_instructors):
        # ค้นหาวิชาที่อาจารย์คนนี้สอน
        taught_courses_indices = np.where(chromosome == inst_id)[0]

        if len(taught_courses_indices) > 1:
            # คัดกรองเฉพาะวิชาบังคับ (type=1)
            compulsory_courses_info = []
            for course_idx in taught_courses_indices:
                if course_year_semester_type[course_idx, 2] == 1: # courseType == 1
                    # เก็บ (year, semester)
                    compulsory_courses_info.append(
                        (course_year_semester_type[course_idx, 0], course_year_semester_type[course_idx, 1])
                    )

            # ตรวจสอบว่ามี (year, semester) ซ้ำกันหรือไม่
            if len(compulsory_courses_info) > 1:
                # สร้าง set เทียมเพื่อตรวจสอบค่าซ้ำ
                unique_infos = []
                for info in compulsory_courses_info:
                    is_unique = True
                    for unique_info in unique_infos:
                        if info[0] == unique_info[0] and info[1] == unique_info[1]:
                            is_unique = False
                            break
                    if is_unique:
                        unique_infos.append(info)

                if len(unique_infos) < len(compulsory_courses_info):
                    fn2 = 100000
                    break # เจอคนเดียวก็พอ

    # --- FN3: จำนวนวิชาที่ไม่อยากสอน (-1) ---
    fn3 = 0
    for course_idx in range(num_courses):
        instructor_id = chromosome[course_idx]
        if preference_matrix[instructor_id, course_idx] == -1:
            fn3 += 1

    # --- FN4 & FN5: SD และ Mean ของวิชาที่อยากสอน (1) ---
    wished_courses_counts = np.zeros(num_instructors)
    for course_idx in range(num_courses):
        instructor_id = chromosome[course_idx]
        if preference_matrix[instructor_id, course_idx] == 1:
            wished_courses_counts[instructor_id] += 1

    fn4 = np.std(wished_courses_counts)
    fn5 = np.mean(wished_courses_counts)

    # --- FN6: จำนวนอาจารย์ที่ assigned credit ต่างจาก personal_credits เกิน 0.5 ---
    # Adjusted FN6 condition: assigned credit > personal_credit + 1 OR assigned credit < personal_credit - 0.5
    fn6 = np.sum((assigned_credits_per_instructor < instructor_personal_credits - 0.5))


    return fn1 + fn2 + fn3*100 + (fn4 - fn5)*200 + fn6 * 1000

# Kernel สำหรับ CUDA ที่จะรันบน GPU
# แต่ละ thread ของ GPU จะคำนวณ fitness ของ chromosome หนึ่งตัว
if HAS_GPU:
    @numba.cuda.jit
    def fitness_cuda_kernel(population, fitness_values, course_credits, course_year_semester_type,
                            instructor_personal_credits, preference_matrix):
        idx = numba.cuda.grid(1)
        if idx < population.shape[0]:
            chromosome = population[idx]
            fitness_values[idx] = calculate_fitness_for_chromosome(
                chromosome,
                course_credits,
                course_year_semester_type,
                instructor_personal_credits,
                preference_matrix
            )

# ============== 3. คลาส GeneticScheduler ==============
class GeneticScheduler:
    def __init__(self, courses_df, instructors_df, population_size=500, num_generations=100,
                 num_elites=50, mutation_rate=0.1):
        self.courses_df = courses_df
        self.instructors_df = instructors_df
        self.initial_population_size = population_size # Store initial size
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_elites = num_elites
        self.mutation_rate = mutation_rate
        self.stagnation_threshold = 5 # Number of generations with no fitness improvement to trigger population increase
        self.stagnation_counter = 0 # Counter for fitness stagnation

        self.num_courses = len(courses_df)
        self.num_instructors = len(instructors_df)

        # --- แปลง DataFrame เป็น NumPy Arrays เพื่อประสิทธิภาพ ---
        self.course_credits = courses_df['courseCredit'].to_numpy()
        self.course_year_semester_type = courses_df[['year', 'semester', 'courseType']].to_numpy()
        self.instructor_personal_credits = instructors_df['personalCredit'].to_numpy()

        pref_cols = courses_df['courseID'].tolist()
        self.preference_matrix = instructors_df[pref_cols].to_numpy()

        self.population = self._create_initial_population()
        self.best_fitness_ever = float('inf') # Track the best fitness found so far


    def _create_initial_population(self):
        """สร้าง gene pool เริ่มต้น โดย 60% จะสุ่มจากอาจารย์ที่ preference เป็น 1"""
        population = []
        num_preferred_chromosomes = int(self.population_size * 0.6)

        # สร้าง Chromosome ที่สุ่มจากอาจารย์ที่ Preference เป็น 1 (ถ้ามี)
        for _ in range(num_preferred_chromosomes):
            chromosome = np.zeros(self.num_courses, dtype=int)
            for course_idx in range(self.num_courses):
                # ค้นหาอาจารย์ที่ Preference เป็น 1 สำหรับวิชานี้
                preferred_instructors = np.where(self.preference_matrix[:, course_idx] == 1)[0]

                if len(preferred_instructors) > 0:
                    # สุ่มเลือกอาจารย์จากกลุ่มที่ Preference เป็น 1
                    assigned_instructor = np.random.choice(preferred_instructors)
                else:
                    # ถ้าไม่มีอาจารย์ที่ Preference เป็น 1 เลย ให้สุ่มเลือกใครก็ได้
                    assigned_instructor = np.random.randint(0, self.num_instructors)
                chromosome[course_idx] = assigned_instructor
            population.append(chromosome)

        # สร้าง Chromosome ที่เหลือแบบสุ่มปกติ
        for _ in range(self.population_size - num_preferred_chromosomes):
            chromosome = np.random.randint(0, self.num_instructors, size=self.num_courses)
            population.append(chromosome)

        return np.array(population)


    def _calculate_population_fitness(self):
        """คำนวณ Fitness ของประชากรทั้งหมด โดยเลือกใช้ GPU หรือ CPU"""
        fitness_values = np.zeros(self.population_size)

        if HAS_GPU:
            # --- GPU Path ---
            # 1. ย้ายข้อมูลไปยังหน่วยความจำ GPU
            d_population = numba.cuda.to_device(self.population)
            d_fitness_values = numba.cuda.to_device(fitness_values)
            d_course_credits = numba.cuda.to_device(self.course_credits)
            d_course_yst = numba.cuda.to_device(self.course_year_semester_type)
            d_inst_credits = numba.cuda.to_device(self.instructor_personal_credits)
            d_pref_matrix = numba.cuda.to_device(self.preference_matrix)

            # 2. กำหนดขนาดของ grid และ block สำหรับ CUDA
            threads_per_block = 256
            blocks_per_grid = (self.population_size + (threads_per_block - 1)) // threads_per_block

            # 3. เรียกใช้ Kernel
            fitness_cuda_kernel[blocks_per_grid, threads_per_block](
                d_population, d_fitness_values, d_course_credits, d_course_yst,
                d_inst_credits, d_pref_matrix
            )

            # 4. คัดลอกผลลัพธ์กลับมายัง CPU
            fitness_values = d_fitness_values.copy_to_host()
        else:
            # --- CPU Path (Parallel using Joblib) ---
            # ใช้ Joblib เพื่อรัน `calculate_fitness_for_chromosome` แบบขนานบนหลายๆ CPU core
            # backend="threading" อาจเร็วกว่าสำหรับฟังก์ชันที่ถูก JIT compile แล้ว เพราะ GIL ถูกปล่อย
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(calculate_fitness_for_chromosome)(
                    self.population[i],
                    self.course_credits,
                    self.course_year_semester_type,
                    self.instructor_personal_credits,
                    self.preference_matrix
                ) for i in range(self.population_size)
            )
            fitness_values = np.array(results)

        return fitness_values

    def _selection(self, fitness_values):
        """คัดเลือก Elites (Chromosome ที่ดีที่สุด)"""
        # เรียงลำดับ index ตามค่า fitness จากน้อยไปมาก
        sorted_indices = np.argsort(fitness_values)
        elite_indices = sorted_indices[:self.num_elites]
        return self.population[elite_indices]

    def _crossover(self, parent1, parent2):
        """ทำการ Crossover เพื่อสร้างลูก"""
        crossover_point = np.random.randint(1, self.num_courses - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def _mutate(self, chromosome):
        """ทำการ Mutation"""
        mutated_chromosome = np.copy(chromosome)
        num_mutations = np.random.randint(1, 11) # สุ่มจำนวน mutation ระหว่าง 1 ถึง 10

        for _ in range(num_mutations):
            if np.random.rand() < self.mutation_rate: # ใช้ mutation_rate เป็นโอกาสที่จะทำการ mutation ในตำแหน่งหนึ่งๆ
                # เลือกประเภท mutation: 50% change, 50% swap
                if np.random.rand() < 0.5:
                    # Change mutation
                    mutation_point = np.random.randint(0, self.num_courses)
                    mutated_chromosome[mutation_point] = np.random.randint(0, self.num_instructors)
                else:
                    # Swap mutation
                    swap_point1 = np.random.randint(0, self.num_courses)
                    swap_point2 = np.random.randint(0, self.num_courses)
                    mutated_chromosome[swap_point1], mutated_chromosome[swap_point2] = mutated_chromosome[swap_point2], mutated_chromosome[swap_point1]
        return mutated_chromosome

    def run(self):
        """เริ่มกระบวนการ Genetic Algorithm"""
        best_fitness_history = []

        for generation in tqdm(range(self.num_generations), desc="Evolving"):
            fitness_values = self._calculate_population_fitness()

            current_best_fitness = np.min(fitness_values)
            best_fitness_history.append(current_best_fitness)

            # Check for fitness stagnation
            if current_best_fitness < self.best_fitness_ever:
                self.best_fitness_ever = current_best_fitness
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1


            # Report fitness every 10 generations
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}: Best Fitness = {best_fitness_history[-1]:.2f}")

            # Report detailed metrics every 100 generations
            if (generation + 1) % 50 == 0 or generation == self.num_generations - 1:
                print(f"\n========== Detailed Metrics at Generation {generation + 1} ==========")
                # Get the best chromosome from the current generation
                best_chromosome_current_gen = self.population[np.argmin(fitness_values)]

                wished_courses_counts = np.zeros(self.num_instructors)
                unwished_courses_assigned_total = 0

                # Recalculate assigned credits for accurate reporting
                assigned_credits_per_instructor = np.zeros(self.num_instructors)
                for course_idx, instructor_id in enumerate(best_chromosome_current_gen):
                    assigned_credits_per_instructor[instructor_id] += self.course_credits[course_idx]
                    preference = self.preference_matrix[instructor_id, course_idx]
                    if preference == 1:
                        wished_courses_counts[instructor_id] += 1
                    elif preference == -1:
                        unwished_courses_assigned_total += 1

                avg_wished_assigned = np.mean(wished_courses_counts)
                sd_wished_assigned = np.std(wished_courses_counts)
                fn6_value = np.sum((assigned_credits_per_instructor < self.instructor_personal_credits - 0.5))


                print(f"ค่าเฉลี่ยจำนวนวิชาที่อาจารย์มี preference score เท่ากับ 1 และได้รับ assign: {avg_wished_assigned:.4f}")
                print(f"SD จำนวนวิชาที่อาจารย์มี preference score เท่ากับ 1 และได้รับ assign: {sd_wished_assigned:.4f}")
                print(f"ผลรวมจำนวนวิชาที่อาจารย์มี preference score เท่ากับ -1 และได้รับ assign: {unwished_courses_assigned_total}")
                print(f"จำนวนอาจารย์ที่ Assigned Credit ต่างกับ Personal Credit เกิน 0.5: {fn6_value}") # This label is kept for consistency, but the condition is changed
                print("=" * 40) # Separator for clarity


            # 1. Selection
            elites = self._selection(fitness_values)

            # 2. สร้าง Generation ใหม่
            next_generation = []
            next_generation.extend(elites) # นำ Elites ไปยัง Generation ถัดไปเลย

            # 3. Crossover & Mutation
            while len(next_generation) < self.population_size:
                # สุ่มเลือกพ่อแม่จาก Elites เพื่อให้ได้ลูกที่ดี
                parent1, parent2 = elites[np.random.choice(len(elites), 2, replace=False)]
                child1, child2 = self._crossover(parent1, parent2)

                next_generation.append(self._mutate(child1))
                if len(next_generation) < self.population_size:
                    next_generation.append(self._mutate(child2))

            self.population = np.array(next_generation)

        # คำนวณ Fitness ของ Generation สุดท้ายและหาตัวที่ดีที่สุด
        final_fitness = self._calculate_population_fitness()
        sorted_indices = np.argsort(final_fitness)

        self.best_chromosomes = self.population[sorted_indices]
        self.best_fitnesses = final_fitness[sorted_indices]

        print("\nEvolution complete.")
        print(f"Best fitness found: {self.best_fitnesses[0]:.2f}")

    def report_results(self):
        """รายงานผลลัพธ์"""
        if not hasattr(self, 'best_chromosomes'):
            print("Please run the algorithm first.")
            return

        # --- แสดงผลสำหรับ 2 อันดับแรก ---
        for i in range(2):
            print(f"\n========== Reporting for Solution #{i+1} (Fitness: {self.best_fitnesses[i]:.2f}) ==========")
            best_chromosome = self.best_chromosomes[i]

            # สร้าง DataFrame ผลลัพธ์
            report_df = self.instructors_df[['InstructorID', 'name', 'personalCredit']].copy()

            # สร้างตาราง assignment matrix (1=assigned, 0=not assigned)
            assignment_matrix = np.zeros((self.num_instructors, self.num_courses), dtype=int)
            assigned_credits_per_instructor = np.zeros(self.num_instructors) # สำหรับ FN6 ใน Report

            for course_idx, instructor_id in enumerate(best_chromosome):
                assignment_matrix[instructor_id, course_idx] = 1
                assigned_credits_per_instructor[instructor_id] += self.course_credits[course_idx]


            assignment_df = pd.DataFrame(assignment_matrix, columns=self.courses_df['courseID'])

            final_report_df = pd.concat([report_df, assignment_df, ], axis=1)

            print("Assignment Table (1 = Assigned):")
            # แสดงผลแค่บางส่วนเพื่อไม่ให้ยาวเกินไป
            display_cols = ['InstructorID', 'name'] + self.courses_df['courseID'].tolist()[:10]
            display(final_report_df[display_cols].head())


            # เพิ่มการแสดง Assigned Credit และ Personal Credit และผลต่าง
            print("\nAssigned vs Personal Credits:")
            credit_summary_df = pd.DataFrame({
                'InstructorID': self.instructors_df['InstructorID'],
                'name': self.instructors_df['name'],
                'PersonalCredit': self.instructor_personal_credits,
                'AssignedCredit': assigned_credits_per_instructor,
                'CreditDifference': np.abs(assigned_credits_per_instructor - self.instructor_personal_credits)
            })
            display(credit_summary_df.head())


        # --- คำนวณค่าสถิติตามที่ต้องการสำหรับโซลูชันที่ดีที่สุด ---
        print("\n========== Final Metrics for the Best Solution (#1) ==========")
        best_solution = self.best_chromosomes[0]

        wished_courses_counts = np.zeros(self.num_instructors)
        unwished_courses_assigned_total = 0
        assigned_credits_per_instructor = np.zeros(self.num_instructors)

        for course_idx, instructor_id in enumerate(best_solution):
            assigned_credits_per_instructor[instructor_id] += self.course_credits[course_idx]
            preference = self.preference_matrix[instructor_id, course_idx]
            if preference == 1:
                wished_courses_counts[instructor_id] += 1
            elif preference == -1:
                unwished_courses_assigned_total += 1

        avg_wished_assigned = np.mean(wished_courses_counts)
        sd_wished_assigned = np.std(wished_courses_counts)
        fn6_value = np.sum((assigned_credits_per_instructor < self.instructor_personal_credits - 0.5))


        print(f"ค่าเฉลี่ยจำนวนวิชาที่อาจารย์มี preference score เท่ากับ 1 และได้รับ assign: {avg_wished_assigned:.4f}")
        print(f"SD จำนวนวิชาที่อาจารย์มี preference score เท่ากับ 1 และได้รับ assign: {sd_wished_assigned:.4f}")
        print(f"ผลรวมจำนวนวิชาที่อาจารย์มี preference score เท่ากับ -1 และได้รับ assign: {unwished_courses_assigned_total}")
        print(f"จำนวนอาจารย์ที่ Assigned Credit ต่างกับ Personal Credit เกิน 0.5: {fn6_value}") # This label is kept for consistency, but the condition is changed


# ============== 4. การทำงานหลัก ==============
if __name__ == '__main__':
    # สร้างข้อมูล
    courses, instructors = create_sample_data(num_courses=140, num_instructors=40)

    print("--- Course Data Sample ---")
    display(courses.head())
    print("\n--- Instructor Data Sample ---")
    display(instructors.iloc[:, :10].head()) # แสดงแค่ 10 วิชาแรก
    print(f"\nTotal Course Credits: {courses['courseCredit'].sum()}")
    print(f"Total Adjusted Personal Credits: {instructors['personalCredit'].sum():.2f}")


    # สร้างและรัน Scheduler 10 ครั้ง
    num_runs = 10
    for run_num in range(1, num_runs + 1):
        print(f"\n========== Running Genetic Algorithm - Run {run_num}/{num_runs} ==========")
        scheduler = GeneticScheduler(
            courses_df=courses,
            instructors_df=instructors,
            population_size=4000,
            num_generations=600,
            num_elites=50,
            mutation_rate=0.05
        )

        scheduler.run()

        # แสดงผลลัพธ์สำหรับ run นี้
        scheduler.report_results()