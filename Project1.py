import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Sensor:
    """Class for sensors that generate data to be collected by Alice."""
    def __init__(self, position):
        self.position = position  # (x, y) position of the sensor
        self.data = random.random()  # Simulated sensor data
    
    def generate_data(self):
        """Simulate data generation at the sensor."""
        self.data = random.random()  # New data is generated each time

class Alice:
    def __init__(self, id, position, power=1.0):
        self.id = id  # Alice's ID
        self.position = position  # (x, y) position
        self.power = power  # Transmission power
        self.last_sent_time = 0  # Time of last transmission
        self.aoi = 0  # Age of Information
        self.collected_data = None
    
    def collect_data(self, sensors):
        """Collect data from nearby sensors.""" 
        for sensor in sensors:
            # Simple Euclidean distance to check if Alice is close enough to the sensor
            distance = np.linalg.norm(np.array(self.position) - np.array(sensor.position))
            if distance < 15:  # Assuming Alice can collect data within a 15-unit range
                self.collected_data = sensor.data
                sensor.generate_data()  # Simulate new data generation by the sensor
                break  # Alice can collect data from only one sensor at a time

    def update_aoi(self, current_time):
        """Updates Age of Information.""" 
        self.aoi = current_time - self.last_sent_time

    def send_data(self, bob, current_time):
        """Send data to Bob and update AoI.""" 
        if self.collected_data is not None:
            bob.receive_data(self.collected_data, current_time)  # Bob receives the data
            self.last_sent_time = current_time
            self.update_aoi(current_time)

    def adjust_power(self, new_power):
        """Adjust transmission power.""" 
        self.power = new_power

class Bob:
    """Class for Bob, the receiver of the data from Alice.""" 
    def __init__(self, position):
        self.position = position  # (x, y) position of Bob
        self.received_data = None  # Latest data received
        self.last_received_time = 0  # Time when last data was received
    
    def receive_data(self, data, current_time):
        """Receive data from Alice.""" 
        self.received_data = data
        self.last_received_time = current_time

    def get_aoi(self, current_time):
        """Get Age of Information (AoI) at Bob.""" 
        return current_time - self.last_received_time

class Warden:
    def __init__(self, position):
        self.position = position  # Warden's position

    def detect(self, alice, detection_threshold=0.5):
        """Detect Alice's transmission based on power and distance.""" 
        distance = np.linalg.norm(np.array(alice.position) - np.array(self.position))
        detection_prob = np.exp(-distance / (alice.power + 1))  # Simplified model
        return detection_prob > detection_threshold

class Environment:
    
    def __init__(self, map_size, num_alices, num_sensors, num_bobs, num_wardens, seed=42):
        self.map_size = map_size
        self.num_alices = num_alices
        self.num_sensors = num_sensors
        self.num_bobs = num_bobs
        self.num_wardens = num_wardens
        self.current_time = 0  # Time step for simulation
        
        # Set the random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Distribute Alices evenly across the grid within (25, 25) to (75, 75)
        self.alices = self.distribute_alices_evenly(num_alices)
        
        # Distribute Sensors randomly within the grid (total of 20), within (25, 25) to (75, 75)
        self.sensors = self.distribute_sensors_randomly()
        
        # Distribute Bobs inside the range of Alices, within (25, 25) to (75, 75)
        self.bobs = self.distribute_bobs_in_range_of_alices()
        
        # Distribute Wardens inside the range of Alices, within (25, 25) to (75, 75)
        self.wardens = self.distribute_wardens_in_range_of_alices()

    def check_collision(self, new_position, all_entities, min_distance=5):
        """Check if the new position collides with any existing entity."""
        for entity in all_entities:
            distance = np.linalg.norm(np.array(new_position) - np.array(entity.position))
            if distance < min_distance:
                return True
        return False

    def distribute_alices_evenly(self, num_alices):
        """Distribute Alices evenly across the grid within (25, 25) to (75, 75)."""
        alices = []
        positions = [
            (35, 35), (35, 65), (65, 35), (65, 65)
        ]
        
        for i, pos in enumerate(positions[:num_alices]):
            alices.append(Alice(i + 1, pos))  # Alice's ID starts at 1
        return alices

    def distribute_bobs_in_range_of_alices(self):
        """Distribute Bobs such that they are within the transmission range of at least one Alice."""
        bobs = []
        all_entities = self.alices  # Start by considering Alices
        for i, alice in enumerate(self.alices):
            # Place Bob near Alice's range (within 15 units), ensuring no collision
            placed = False
            while not placed:
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(0, 15)  # Bob is within Alice's transmission range
                x = alice.position[0] + distance * np.cos(angle)
                y = alice.position[1] + distance * np.sin(angle)
                # Ensure Bob is inside the (25, 25) to (75, 75) bounds
                x = np.clip(x, 25, 75)
                y = np.clip(y, 25, 75)
                new_position = (x, y)
                if not self.check_collision(new_position, all_entities):
                    bobs.append(Bob(new_position))
                    all_entities.append(Bob(new_position))  # Add Bob to the checked entities
                    placed = True
        return bobs

    def distribute_wardens_in_range_of_alices(self):
        """Distribute Wardens near Alices within range (15 units) and within (25, 25) to (75, 75)."""
        wardens = []
        all_entities = self.alices + self.bobs  # Start by considering Alices and Bobs
        for i, alice in enumerate(self.alices):
            # Place Warden near Alice's range (within 15 units), ensuring no collision
            placed = False
            while not placed:
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(0, 15)  # Warden is within Alice's transmission range
                x = alice.position[0] + distance * np.cos(angle)
                y = alice.position[1] + distance * np.sin(angle)
                # Ensure Warden is inside the (25, 25) to (75, 75) bounds
                x = np.clip(x, 25, 75)
                y = np.clip(y, 25, 75)
                new_position = (x, y)
                if not self.check_collision(new_position, all_entities):
                    wardens.append(Warden(new_position))
                    all_entities.append(Warden(new_position))  # Add Warden to the checked entities
                    placed = True
        return wardens

    def distribute_sensors_randomly(self):
        """Distribute sensors randomly within the grid, within (25, 25) to (75, 75)."""
        sensors = []
        all_entities = self.alices + self.bobs + self.wardens  # Consider all Alices, Bobs, and Wardens
        for _ in range(self.num_sensors):
            placed = False
            while not placed:
                x = random.randint(25, 75)
                y = random.randint(25, 75)
                new_position = (x, y)
                if not self.check_collision(new_position, all_entities):
                    sensors.append(Sensor(new_position))
                    all_entities.append(Sensor(new_position))  # Add Sensor to the checked entities
                    placed = True
        return sensors

    def step(self, actions):
        """Advance the simulation by one time step.""" 
        total_reward = 0
        for i, alice in enumerate(self.alices):
            # Each Alice collects data from sensors
            alice.collect_data(self.sensors)

            # Each Alice sends the collected data to Bob
            bob = self.bobs[i % len(self.bobs)]  # Simple assignment of Bob
            alice.adjust_power(actions[i])  # Apply power control action
            alice.send_data(bob, self.current_time)

            # Calculate AoI and detection penalty for each Alice
            aoi_penalty = alice.aoi  # Older data incurs a penalty
            detection_penalty = 1 if any([warden.detect(alice) for warden in self.wardens]) else 0
            reward = - (aoi_penalty + 2 * detection_penalty)  # Adjust reward function
            total_reward += reward

            # Increment time
            self.current_time += 1

        return total_reward

    def render(self):
        """Render the environment with Alices, Bobs, Wardens, and Sensors.""" 
        plt.figure(figsize=(10, 10))
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        
        # Plot Alices (UAVs)
        for alice in self.alices:
            plt.scatter(alice.position[0], alice.position[1], color='blue', s=100, label="Alice" if alice == self.alices[0] else "")
            circle = plt.Circle(alice.position, 15, color='blue', fill=False, linestyle='dotted', linewidth=1)
            plt.gca().add_patch(circle)
            plt.text(alice.position[0], alice.position[1] + 1, str(alice.id), color='blue', ha='center', fontsize=12)

        # Plot Bobs (receivers)
        for bob in self.bobs:
            plt.scatter(bob.position[0], bob.position[1], color='green', s=100, label="Bob" if bob == self.bobs[0] else "")
            plt.text(bob.position[0], bob.position[1] + 1, str(self.bobs.index(bob) + 1), color='green', ha='center', fontsize=12)

        # Plot Wardens (detection agents)
        for warden in self.wardens:
            plt.scatter(warden.position[0], warden.position[1], color='red', s=100, marker='x', label="Warden" if warden == self.wardens[0] else "")
            circle = plt.Circle(warden.position, 15, color='red', fill=False, linestyle='dotted', linewidth=1)
            plt.gca().add_patch(circle)
            plt.text(warden.position[0], warden.position[1] + 1, str(self.wardens.index(warden) + 1), color='red', ha='center', fontsize=12)

        # Plot Sensors
        for sensor in self.sensors:
            plt.scatter(sensor.position[0], sensor.position[1], color='purple', s=50, marker='.', label="Sensor" if sensor == self.sensors[0] else "")
        
        # Display legend and grid
        plt.title("Simulation Environment")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

# Create the environment with 4 Alices, Bobs, and Wardens and a fixed random seed
env = Environment(map_size=100, num_alices=4, num_sensors=20, num_bobs=4, num_wardens=4, seed=42)
env.render()  # Display the environment grid
