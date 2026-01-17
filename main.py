import mesa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import random
import math


class SmokeParticle(Agent):
    """
    A smoke particle agent with fluid dynamics properties.
    Each particle has velocity, density, temperature, and position.
    """

    def __init__(self, unique_id, model, pos, density=1.0, temperature=1.0):
        super().__init__(unique_id, model)
        self.pos = pos  # (x, y) position
        self.velocity = [0.0, 0.0]  # [vx, vy] velocity
        self.density = density  # smoke density
        self.temperature = temperature  # temperature affects buoyancy
        self.age = 0  # particle age for decay
        self.max_age = 100  # maximum lifetime

    def step(self):
        """
        Execute one step of the smoke particle behavior.
        """
        self.age += 1

        # Apply fluid dynamics forces
        self.apply_buoyancy()
        self.apply_diffusion()
        self.apply_advection()
        self.apply_decay()

        # Update position based on velocity
        self.update_position()

        # Remove old particles
        if self.age > self.max_age or self.density < 0.01:
            self.model.schedule.remove(self)

    def apply_buoyancy(self):
        """
        Apply buoyancy force - hot smoke rises.
        """
        buoyancy_force = self.temperature * self.model.gravity * self.model.buoyancy_strength
        self.velocity[1] += buoyancy_force

    def apply_diffusion(self):
        """
        Apply diffusion - smoke spreads out due to concentration gradients.
        """
        # Get nearby particles for diffusion calculation
        neighbors = self.model.space.get_neighbors(
            self.pos, self.model.diffusion_radius, include_center=False
        )

        if neighbors:
            # Calculate average density of neighbors
            neighbor_densities = [n.density for n in neighbors if isinstance(n, SmokeParticle)]
            if neighbor_densities:
                avg_neighbor_density = sum(neighbor_densities) / len(neighbor_densities)

                # Diffusion force based on density gradient
                density_diff = avg_neighbor_density - self.density
                diffusion_force = density_diff * self.model.diffusion_strength

                # Apply random diffusion direction
                angle = random.uniform(0, 2 * math.pi)
                self.velocity[0] += diffusion_force * math.cos(angle)
                self.velocity[1] += diffusion_force * math.sin(angle)

    def apply_advection(self):
        """
        Apply advection - particles move with the flow field.
        """
        # Add environmental wind/flow
        self.velocity[0] += self.model.wind_force[0]
        self.velocity[1] += self.model.wind_force[1]

        # Add some turbulence
        turbulence_x = random.uniform(-self.model.turbulence, self.model.turbulence)
        turbulence_y = random.uniform(-self.model.turbulence, self.model.turbulence)
        self.velocity[0] += turbulence_x
        self.velocity[1] += turbulence_y

    def apply_decay(self):
        """
        Apply decay and cooling over time.
        """
        # Density decay
        self.density *= 0.995

        # Temperature cooling
        self.temperature *= 0.998

        # Velocity damping (friction)
        self.velocity[0] *= 0.98
        self.velocity[1] *= 0.98

    def update_position(self):
        """
        Update position based on current velocity.
        """
        new_x = self.pos[0] + self.velocity[0] * self.model.dt
        new_y = self.pos[1] + self.velocity[1] * self.model.dt

        # Boundary conditions - reflect at walls
        if new_x < 0 or new_x > self.model.width:
            self.velocity[0] *= -0.5
            new_x = max(0, min(self.model.width, new_x))

        if new_y < 0 or new_y > self.model.height:
            self.velocity[1] *= -0.5
            new_y = max(0, min(self.model.height, new_y))

        # Update position in space
        self.model.space.move_agent(self, (new_x, new_y))


class FluidDynamicsModel(Model):
    """
    A model simulating 2D fluid dynamics with smoke particles.
    """

    def __init__(self, width=50, height=50, initial_particles=200):
        super().__init__()

        # Model parameters
        self.width = width
        self.height = height
        self.dt = 0.1  # time step

        # Physical parameters
        self.gravity = -0.01  # gravity strength
        self.buoyancy_strength = 0.05  # buoyancy effect
        self.diffusion_strength = 0.02  # diffusion rate
        self.diffusion_radius = 3.0  # diffusion interaction radius
        self.turbulence = 0.001  # turbulence strength
        self.wind_force = [0.005, 0.0]  # environmental wind [x, y]

        # Emission parameters
        self.emission_rate = 5  # particles emitted per step
        self.emission_pos = [width * 0.2, height * 0.1]  # emission source position
        self.emission_temperature = 2.0  # initial temperature of emitted particles
        self.emission_density = 1.0  # initial density of emitted particles

        # Setup
        self.space = ContinuousSpace(width, height, torus=False)
        self.schedule = RandomActivation(self)

        # Create initial particles
        self.create_initial_particles(initial_particles)

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Particles": lambda m: len(m.schedule.agents),
                "Average_Density": lambda m: np.mean(
                    [a.density for a in m.schedule.agents]) if m.schedule.agents else 0,
                "Average_Temperature": lambda m: np.mean(
                    [a.temperature for a in m.schedule.agents]) if m.schedule.agents else 0,
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def create_initial_particles(self, num_particles):
        """
        Create initial smoke particles near the emission source.
        """
        for i in range(num_particles):
            # Random position near emission source
            x = self.emission_pos[0] + random.uniform(-2, 2)
            y = self.emission_pos[1] + random.uniform(-1, 1)

            # Ensure within bounds
            x = max(0, min(self.width, x))
            y = max(0, min(self.height, y))

            # Create particle
            particle = SmokeParticle(
                i, self, (x, y),
                density=random.uniform(0.5, 1.0),
                temperature=random.uniform(1.0, 2.0)
            )

            self.space.place_agent(particle, (x, y))
            self.schedule.add(particle)

    def emit_particles(self):
        """
        Continuously emit new smoke particles from the source.
        """
        for _ in range(self.emission_rate):
            # Random position near emission source
            x = self.emission_pos[0] + random.uniform(-1, 1)
            y = self.emission_pos[1] + random.uniform(-0.5, 0.5)

            # Ensure within bounds
            x = max(0, min(self.width, x))
            y = max(0, min(self.height, y))

            # Create new particle with unique ID
            particle_id = len(self.schedule.agents) + random.randint(1000, 9999)
            particle = SmokeParticle(
                particle_id, self, (x, y),
                density=self.emission_density,
                temperature=self.emission_temperature
            )

            # Give initial upward velocity
            particle.velocity = [random.uniform(-0.01, 0.01), random.uniform(0.01, 0.03)]

            self.space.place_agent(particle, (x, y))
            self.schedule.add(particle)

    def step(self):
        """
        Execute one step of the model.
        """
        # Emit new particles
        self.emit_particles()

        # Step all agents
        self.schedule.step()

        # Collect data
        self.datacollector.collect(self)


class FluidVisualizer:
    """
    Visualizer for the fluid dynamics simulation.
    """

    def __init__(self, model):
        self.model = model

    def create_density_field(self, grid_size=20):
        """
        Create a density field for visualization.
        """
        x_bins = np.linspace(0, self.model.width, grid_size)
        y_bins = np.linspace(0, self.model.height, grid_size)
        density_field = np.zeros((grid_size - 1, grid_size - 1))

        for agent in self.model.schedule.agents:
            if isinstance(agent, SmokeParticle):
                x_idx = min(int(agent.pos[0] / self.model.width * (grid_size - 1)), grid_size - 2)
                y_idx = min(int(agent.pos[1] / self.model.height * (grid_size - 1)), grid_size - 2)
                density_field[y_idx, x_idx] += agent.density

        return density_field

    def create_velocity_field(self, grid_size=15):
        """
        Create velocity field vectors for visualization.
        """
        x_positions = np.linspace(0, self.model.width, grid_size)
        y_positions = np.linspace(0, self.model.height, grid_size)
        u_field = np.zeros((grid_size, grid_size))
        v_field = np.zeros((grid_size, grid_size))
        count_field = np.zeros((grid_size, grid_size))

        for agent in self.model.schedule.agents:
            if isinstance(agent, SmokeParticle):
                x_idx = min(int(agent.pos[0] / self.model.width * (grid_size - 1)), grid_size - 1)
                y_idx = min(int(agent.pos[1] / self.model.height * (grid_size - 1)), grid_size - 1)

                u_field[y_idx, x_idx] += agent.velocity[0]
                v_field[y_idx, x_idx] += agent.velocity[1]
                count_field[y_idx, x_idx] += 1

        # Average velocities where there are particles
        mask = count_field > 0
        u_field[mask] /= count_field[mask]
        v_field[mask] /= count_field[mask]

        return x_positions, y_positions, u_field, v_field

    def plot_simulation(self, step_num=0, save_fig=False):
        """
        Create a comprehensive visualization of the simulation.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Fluid Dynamics Simulation - Step {step_num}', fontsize=16)

        # 1. Particle positions colored by density
        particles = [agent for agent in self.model.schedule.agents if isinstance(agent, SmokeParticle)]
        if particles:
            x_pos = [p.pos[0] for p in particles]
            y_pos = [p.pos[1] for p in particles]
            densities = [p.density for p in particles]

            scatter = ax1.scatter(x_pos, y_pos, c=densities, cmap='Reds', alpha=0.7, s=20)
            ax1.set_title('Particle Positions (colored by density)')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            plt.colorbar(scatter, ax=ax1, label='Density')

            # Mark emission source
            ax1.plot(self.model.emission_pos[0], self.model.emission_pos[1],
                     'ko', markersize=10, label='Emission Source')
            ax1.legend()

        # 2. Density field heatmap
        density_field = self.create_density_field()
        sns.heatmap(density_field, ax=ax2, cmap='Reds', cbar_kws={'label': 'Density'})
        ax2.set_title('Density Field')
        ax2.set_xlabel('X Grid')
        ax2.set_ylabel('Y Grid')

        # 3. Velocity field
        if particles:
            temperatures = [p.temperature for p in particles]
            scatter = ax3.scatter(x_pos, y_pos, c=temperatures, cmap='coolwarm', alpha=0.7, s=20)
            ax3.set_title('Particle Temperatures')
            ax3.set_xlabel('X Position')
            ax3.set_ylabel('Y Position')
            plt.colorbar(scatter, ax=ax3, label='Temperature')

        # 4. Velocity vectors
        x_pos, y_pos, u_field, v_field = self.create_velocity_field()
        X, Y = np.meshgrid(x_pos, y_pos)
        ax4.quiver(X, Y, u_field, v_field, alpha=0.7)
        ax4.set_title('Velocity Field')
        ax4.set_xlabel('X Position')
        ax4.set_ylabel('Y Position')

        plt.tight_layout()

        if save_fig:
            plt.savefig(f'fluid_simulation_step_{step_num}.png', dpi=150, bbox_inches='tight')

        plt.show()

    def plot_statistics(self):
        """
        Plot simulation statistics over time.
        """
        model_data = self.model.datacollector.get_model_vars_dataframe()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Total particles over time
        ax1.plot(model_data.index, model_data['Total_Particles'], 'b-', linewidth=2)
        ax1.set_title('Total Particles Over Time')
        ax1.set_ylabel('Number of Particles')
        ax1.grid(True, alpha=0.3)

        # Average density over time
        ax2.plot(model_data.index, model_data['Average_Density'], 'r-', linewidth=2)
        ax2.set_title('Average Density Over Time')
        ax2.set_ylabel('Average Density')
        ax2.grid(True, alpha=0.3)

        # Average temperature over time
        ax3.plot(model_data.index, model_data['Average_Temperature'], 'orange', linewidth=2)
        ax3.set_title('Average Temperature Over Time')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Average Temperature')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def run_simulation():
    """
    Main function to run the fluid dynamics simulation.
    """
    print("Starting 2D Fluid Dynamics Simulation...")
    print("Simulating smoke with buoyancy, diffusion, and turbulence\n")

    # Create model
    model = FluidDynamicsModel(width=40, height=60, initial_particles=100)
    visualizer = FluidVisualizer(model)

    # Run simulation
    num_steps = 50
    visualization_interval = 10

    print(f"Running simulation for {num_steps} steps...")
    print(f"Visualization every {visualization_interval} steps\n")

    for step in range(num_steps):
        model.step()

        # Print progress
        if step % 10 == 0:
            num_particles = len(model.schedule.agents)
            avg_density = np.mean([a.density for a in model.schedule.agents]) if model.schedule.agents else 0
            print(f"Step {step}: {num_particles} particles, avg density: {avg_density:.3f}")

        # Visualize periodically
        if step % visualization_interval == 0 or step == num_steps - 1:
            print(f"\nVisualizing step {step}...")
            visualizer.plot_simulation(step_num=step)

    # Final statistics
    print("\nSimulation completed!")
    print("Showing final statistics...")
    visualizer.plot_statistics()

    return model, visualizer


if __name__ == "__main__":
    # Run the simulation
    model, visualizer = run_simulation()

    print("\nSimulation Summary:")
    print(f"Final number of particles: {len(model.schedule.agents)}")

    # Get final statistics
    final_data = model.datacollector.get_model_vars_dataframe().iloc[-1]
    print(f"Final average density: {final_data['Average_Density']:.3f}")
    print(f"Final average temperature: {final_data['Average_Temperature']:.3f}")
