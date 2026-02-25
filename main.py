import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import seaborn as sns

@dataclass
class Agent:
    id: int
    age: int
    household_size: float
    income: float
    income_quintile: int
    employment_status: str
    wealth: float
    internet_access: bool
    education: str
    education_multiplier: float
    study_time_available: float
    learning_capacity: float
    skills: float
    oer_aware: bool
    oer_engaged: bool
    engagement_level: str
    quarters_engaged: int
    signal_type: str
    signal_completed: bool
    quarters_since_job_change: int
    
class OERSimulation:
    def __init__(self, n_agents=10000, n_quarters=40, random_seed=2):
        self.rng = np.random.RandomState(random_seed)
        self.n_agents = n_agents
        self.n_quarters = n_quarters
        self.agents: List[Agent] = []
        self.results = []
        
        self.income_quintiles = [
            (0, 29963),
            (29964, 59533),
            (59534, 96562),
            (96563, 160832),
            (160833, 500000)
        ]
        
        self.education_levels = {
            'less_than_hs': {'pct': 0.08, 'multiplier': 0.75},
            'hs_diploma': {'pct': 0.27, 'multiplier': 1.0},
            'some_college': {'pct': 0.29, 'multiplier': 1.15},
            'bachelors': {'pct': 0.24, 'multiplier': 1.79},
            'graduate': {'pct': 0.12, 'multiplier': 2.28}
        }
        
        self.engagement_levels = {
            'casual': {'pct': 0.50, 'hours_per_week': 2.5, 'months': 1.5},
            'moderate': {'pct': 0.30, 'hours_per_week': 7, 'months': 5},
            'serious': {'pct': 0.15, 'hours_per_week': 13.5, 'months': 10.5},
            'intensive': {'pct': 0.05, 'hours_per_week': 19, 'months': 12}
        }
        
        self.signals = {
            'github': {'hours': 200, 'boost': 0.05},
            'projects': {'hours': 150, 'boost': 0.03},
            'opensource': {'hours': 250, 'boost': 0.10},
            'freelance': {'hours': 300, 'boost': 0.15},
            'certification': {'hours': 80, 'cost': 500, 'boost': 0.25}
        }
        
    def initialize_agents(self):
        for i in range(self.n_agents):
            age = self.rng.uniform(24, 45)
            household_size = max(1, self.rng.normal(2.5, 1.3))
            
            quintile = self.rng.choice([0, 1, 2, 3, 4], p=[0.20, 0.20, 0.20, 0.20, 0.20])
            income_min, income_max = self.income_quintiles[quintile]
            income = self.rng.uniform(income_min, income_max)
            
            employment = self.rng.choice(['employed', 'unemployed', 'not_in_lf'], 
                                        p=[0.95, 0.03, 0.02])
            
            if quintile == 0:
                wealth = self.rng.uniform(0, 1000)
            else:
                wealth = self.rng.exponential(scale=5000 * (quintile + 1))
            
            internet_probs = [0.78, 0.85, 0.90, 0.95, 0.99]
            internet_access = self.rng.random() < internet_probs[quintile]
            
            education_choice = self.rng.choice(
                list(self.education_levels.keys()),
                p=[0.08, 0.27, 0.29, 0.24, 0.12]
            )
            education_multiplier = self.education_levels[education_choice]['multiplier']
            income *= education_multiplier
            
            study_time_base = max(0, self.rng.normal(10, 5))
            study_time = study_time_base * (1 - 0.4 * (household_size - 2.5) / 1.3)
            study_time = study_time * (1 + 0.2 * (quintile - 2) / 2)
            study_time = max(0, study_time)
            
            learning_capacity = np.clip(self.rng.normal(1.0, 0.15), 0.7, 1.3)
            
            oer_aware = self.rng.random() < 0.35
            oer_engaged = False
            if oer_aware and internet_access:
                time_factor = min(study_time / 10, 2.0)
                adoption_prob = 0.15 * time_factor
                adoption_prob = min(adoption_prob, 0.50)
                oer_engaged = self.rng.random() < adoption_prob
            
            engagement_level = 'none'
            if oer_engaged:
                engagement_level = self.rng.choice(
                    list(self.engagement_levels.keys()),
                    p=[0.50, 0.30, 0.15, 0.05]
                )
            
            agent = Agent(
                id=i,
                age=age,
                household_size=household_size,
                income=income,
                income_quintile=quintile,
                employment_status=employment,
                wealth=wealth,
                internet_access=internet_access,
                education=education_choice,
                education_multiplier=education_multiplier,
                study_time_available=study_time,
                learning_capacity=learning_capacity,
                skills=0.0,
                oer_aware=oer_aware,
                oer_engaged=oer_engaged,
                engagement_level=engagement_level,
                quarters_engaged=0,
                signal_type='none',
                signal_completed=False,
                quarters_since_job_change=999
            )
            
            self.agents.append(agent)
    
    def acquire_skills(self, agent: Agent, quarter: int):
        if not agent.oer_engaged or agent.engagement_level == 'none':
            return
        
        hours_per_week = self.engagement_levels[agent.engagement_level]['hours_per_week']
        max_months = self.engagement_levels[agent.engagement_level]['months']
        max_quarters = max_months / 3
        
        if agent.quarters_engaged >= max_quarters:
            agent.oer_engaged = False
            return
        
        hours_per_quarter = hours_per_week * 13
        actual_hours = min(hours_per_quarter, agent.study_time_available * 13)
        
        efficiency = 1.0 - (0.005 * agent.skills)
        material_quality = 1.0
        
        skill_gain = (actual_hours / 15) * agent.learning_capacity * efficiency * material_quality
        agent.skills += skill_gain
        agent.quarters_engaged += 1
    
    def select_signal(self, agent: Agent):
        if agent.skills < 40 or agent.signal_completed:
            return
        
        time_available_total = agent.study_time_available * 13
        
        if agent.wealth >= 500 and time_available_total >= 50:
            agent.signal_type = 'certification'
        elif time_available_total >= 250:
            agent.signal_type = 'opensource'
        elif time_available_total >= 200:
            agent.signal_type = 'github'
        elif time_available_total >= 150:
            agent.signal_type = 'projects'
        else:
            agent.signal_type = 'none'
    
    def complete_signal(self, agent: Agent, quarter: int):
        if agent.signal_type == 'none' or agent.signal_completed:
            return
        
        required_hours = self.signals[agent.signal_type]['hours']
        
        if agent.signal_type == 'certification':
            if agent.wealth >= self.signals[agent.signal_type]['cost']:
                agent.wealth -= self.signals[agent.signal_type]['cost']
                agent.signal_completed = True
        else:
            available_hours = agent.study_time_available * 13
            if available_hours >= required_hours:
                agent.signal_completed = True
    
    def calculate_job_change_probability(self, agent: Agent):
        if agent.employment_status == 'unemployed':
            return 0.45
        
        base_prob = 0.05
        if agent.skills >= 60:
            base_prob += 0.10
        if agent.signal_completed:
            base_prob += 0.08
        
        return base_prob
    
    def update_income(self, agent: Agent, quarter: int):
        if agent.employment_status != 'employed':
            return
        
        growth_rate = 0.03 / 4
        
        if agent.skills < 40:
            skill_premium = 0
        elif agent.skills < 70:
            skill_premium = (agent.skills - 40) * 0.008
        else:
            skill_premium = 0.24 + (agent.skills - 70) * 0.004
        
        transition_boost = 0
        if agent.quarters_since_job_change < 8:
            decay = agent.quarters_since_job_change * 0.05
            transition_boost = max(0, 0.15 - decay)
        
        has_degree = agent.education in ['bachelors', 'graduate']
        if has_degree:
            credential_discount = 0
        elif agent.skills >= 60 and agent.signal_completed:
            credential_discount = 0.05
        elif agent.skills >= 60:
            credential_discount = 0.15
        else:
            credential_discount = 0
        
        random_shock = self.rng.normal(1.0, 0.10)
        
        income_multiplier = (1 + growth_rate + skill_premium + transition_boost - credential_discount)
        agent.income = agent.income * income_multiplier * random_shock
        agent.income = max(0, agent.income)
    
    def update_employment(self, agent: Agent, quarter: int):
        job_change_prob = self.calculate_job_change_probability(agent)
        
        if self.rng.random() < job_change_prob:
            if agent.employment_status == 'unemployed':
                agent.employment_status = 'employed'
                median_income = np.median([self.income_quintiles[i][0] + self.income_quintiles[i][1] 
                                          for i in range(5)]) / 2
                agent.income = median_income * agent.education_multiplier
            else:
                agent.quarters_since_job_change = 0
        else:
            agent.quarters_since_job_change += 1
    
    def get_quintile(self, income: float):
        for i, (low, high) in enumerate(self.income_quintiles):
            if low <= income <= high:
                return i
        return 4
    
    def run_simulation(self):
        print("Initializing agents...")
        self.initialize_agents()
        
        initial_data = []
        for agent in self.agents:
            initial_data.append({
                'agent_id': agent.id,
                'quarter': 0,
                'income': agent.income,
                'quintile': agent.income_quintile,
                'skills': agent.skills,
                'oer_engaged': agent.oer_engaged,
                'signal_completed': agent.signal_completed,
                'employment': agent.employment_status
            })
        self.results.extend(initial_data)
        
        print("Running simulation...")
        for quarter in range(1, self.n_quarters + 1):
            if quarter % 10 == 0:
                print(f"Quarter {quarter}/{self.n_quarters}")
            
            for agent in self.agents:
                self.acquire_skills(agent, quarter)
                self.select_signal(agent)
                self.complete_signal(agent, quarter)
                self.update_employment(agent, quarter)
                self.update_income(agent, quarter)
                
                current_quintile = self.get_quintile(agent.income)
                
                self.results.append({
                    'agent_id': agent.id,
                    'quarter': quarter,
                    'income': agent.income,
                    'quintile': current_quintile,
                    'skills': agent.skills,
                    'oer_engaged': agent.oer_engaged,
                    'signal_completed': agent.signal_completed,
                    'employment': agent.employment_status
                })
        
        print("Simulation complete!")
        return pd.DataFrame(self.results)
    
    def analyze_results(self, df: pd.DataFrame):
        initial_df = df[df['quarter'] == 0]
        final_df = df[df['quarter'] == self.n_quarters]
        
        merged = initial_df.merge(final_df, on='agent_id', suffixes=('_initial', '_final'))
        
        print("\n" + "="*60)
        print("SIMULATION RESULTS SUMMARY")
        print("="*60)
        
        print("\n1. INCOME MOBILITY")
        print("-" * 60)
        mobility = merged[merged['quintile_initial'] != merged['quintile_final']]
        print(f"Agents who changed quintiles: {len(mobility)} ({len(mobility)/len(merged)*100:.1f}%)")
        
        upward = merged[merged['quintile_final'] > merged['quintile_initial']]
        print(f"Upward mobility: {len(upward)} ({len(upward)/len(merged)*100:.1f}%)")
        
        downward = merged[merged['quintile_final'] < merged['quintile_initial']]
        print(f"Downward mobility: {len(downward)} ({len(downward)/len(merged)*100:.1f}%)")
        
        print("\n2. OER IMPACT")
        print("-" * 60)
        oer_users = merged[merged['oer_engaged_initial'] == True]
        non_oer_users = merged[merged['oer_engaged_initial'] == False]
        
        oer_upward = len(oer_users[oer_users['quintile_final'] > oer_users['quintile_initial']])
        non_oer_upward = len(non_oer_users[non_oer_users['quintile_final'] > non_oer_users['quintile_initial']])
        
        print(f"OER users with upward mobility: {oer_upward}/{len(oer_users)} ({oer_upward/len(oer_users)*100:.1f}%)")
        print(f"Non-OER users with upward mobility: {non_oer_upward}/{len(non_oer_users)} ({non_oer_upward/len(non_oer_users)*100:.1f}%)")
        
        print(f"\nAverage income change (OER users): ${oer_users['income_final'].mean() - oer_users['income_initial'].mean():.2f}")
        print(f"Average income change (Non-OER): ${non_oer_users['income_final'].mean() - non_oer_users['income_initial'].mean():.2f}")
        
        print(f"\nAverage skills gained (OER users): {oer_users['skills_final'].mean():.1f} points")
        print(f"Average skills gained (Non-OER): {non_oer_users['skills_final'].mean():.1f} points")
        
        print("\n3. SIGNALING IMPACT")
        print("-" * 60)
        with_signal = merged[merged['signal_completed_final'] == True]
        without_signal = merged[(merged['oer_engaged_initial'] == True) & (merged['signal_completed_final'] == False)]
        
        if len(with_signal) > 0:
            signal_upward = len(with_signal[with_signal['quintile_final'] > with_signal['quintile_initial']])
            print(f"OER users with signals - upward mobility: {signal_upward}/{len(with_signal)} ({signal_upward/len(with_signal)*100:.1f}%)")
            print(f"Average income (with signal): ${with_signal['income_final'].mean():.2f}")
        
        if len(without_signal) > 0:
            no_signal_upward = len(without_signal[without_signal['quintile_final'] > without_signal['quintile_initial']])
            print(f"OER users without signals - upward mobility: {no_signal_upward}/{len(without_signal)} ({no_signal_upward/len(without_signal)*100:.1f}%)")
            print(f"Average income (without signal): ${without_signal['income_final'].mean():.2f}")
        
        print("\n4. BY STARTING QUINTILE")
        print("-" * 60)
        for q in range(5):
            q_agents = merged[merged['quintile_initial'] == q]
            q_upward = len(q_agents[q_agents['quintile_final'] > q_agents['quintile_initial']])
            q_oer = len(q_agents[q_agents['oer_engaged_initial'] == True])
            print(f"Quintile {q+1}: {q_upward}/{len(q_agents)} upward ({q_upward/len(q_agents)*100:.1f}%), OER users: {q_oer} ({q_oer/len(q_agents)*100:.1f}%)")
        
        return merged
    
    def create_visualizations(self, df: pd.DataFrame, merged: pd.DataFrame):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        oer_users = merged[merged['oer_engaged_initial'] == True]
        non_oer_users = merged[merged['oer_engaged_initial'] == False]
        
        quintile_transitions_oer = np.zeros((5, 5))
        for _, row in oer_users.iterrows():
            quintile_transitions_oer[int(row['quintile_initial']), int(row['quintile_final'])] += 1
        
        for i in range(5):
            if quintile_transitions_oer[i].sum() > 0:
                quintile_transitions_oer[i] = quintile_transitions_oer[i] / quintile_transitions_oer[i].sum()
        
        sns.heatmap(quintile_transitions_oer, annot=True, fmt='.2f', cmap='YlGnBu', 
                xticklabels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                yticklabels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                ax=axes[0, 0], cbar_kws={'label': 'Probability'})
        axes[0, 0].set_title('Quintile Transition Matrix - OER Users', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Final Quintile')
        axes[0, 0].set_ylabel('Initial Quintile')
        
        quintile_transitions_non = np.zeros((5, 5))
        for _, row in non_oer_users.iterrows():
            quintile_transitions_non[int(row['quintile_initial']), int(row['quintile_final'])] += 1
        
        for i in range(5):
            if quintile_transitions_non[i].sum() > 0:
                quintile_transitions_non[i] = quintile_transitions_non[i] / quintile_transitions_non[i].sum()
        
        sns.heatmap(quintile_transitions_non, annot=True, fmt='.2f', cmap='OrRd', 
                xticklabels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                yticklabels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                ax=axes[0, 1], cbar_kws={'label': 'Probability'})
        axes[0, 1].set_title('Quintile Transition Matrix - Non-OER Users', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Final Quintile')
        axes[0, 1].set_ylabel('Initial Quintile')
        
        oer_by_quintile = []
        non_oer_by_quintile = []
        
        for q in range(5):
            oer_q = oer_users[oer_users['quintile_initial'] == q]
            non_oer_q = non_oer_users[non_oer_users['quintile_initial'] == q]
            
            if len(oer_q) > 0:
                oer_by_quintile.append((oer_q['quintile_final'] > oer_q['quintile_initial']).mean())
            else:
                oer_by_quintile.append(0)
            
            if len(non_oer_q) > 0:
                non_oer_by_quintile.append((non_oer_q['quintile_final'] > non_oer_q['quintile_initial']).mean())
            else:
                non_oer_by_quintile.append(0)
        
        x = np.arange(5)
        width = 0.35
        
        bars1 = axes[0, 2].bar(x - width/2, non_oer_by_quintile, width, label='Non-OER Users', color='#FF6B6B')
        bars2 = axes[0, 2].bar(x + width/2, oer_by_quintile, width, label='OER Users', color='#4ECDC4')
        
        axes[0, 2].set_title('Upward Mobility by Starting Quintile', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Starting Quintile')
        axes[0, 2].set_ylabel('Proportion with Upward Mobility')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        skills_dist = oer_users['skills_final']
        axes[1, 0].hist(skills_dist, bins=30, edgecolor='black', color='#95E1D3', alpha=0.7)
        axes[1, 0].set_title('Skills Distribution (OER Users)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Skill Points')
        axes[1, 0].set_ylabel('Number of Agents')
        axes[1, 0].axvline(40, color='red', linestyle='--', linewidth=2, label='Threshold (40)')
        axes[1, 0].axvline(60, color='orange', linestyle='--', linewidth=2, label='Signal Eligible (60)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        income_change_oer = oer_users['income_final'] - oer_users['income_initial']
        income_change_non = non_oer_users['income_final'] - non_oer_users['income_initial']
        
        summary_text = f"""
    Statistics Summary

    Total Agents: {len(merged):,}
    OER Users: {len(oer_users):,} ({len(oer_users)/len(merged)*100:.1f}%)
    Non-OER: {len(non_oer_users):,} ({len(non_oer_users)/len(merged)*100:.1f}%)

    UPWARD MOBILITY:
    OER: {(oer_users['quintile_final'] > oer_users['quintile_initial']).sum()}/{len(oer_users)} ({(oer_users['quintile_final'] > oer_users['quintile_initial']).mean()*100:.1f}%)
    Non-OER: {(non_oer_users['quintile_final'] > non_oer_users['quintile_initial']).sum()}/{len(non_oer_users)} ({(non_oer_users['quintile_final'] > non_oer_users['quintile_initial']).mean()*100:.1f}%)

    MEDIAN INCOME CHANGE:
    OER: ${income_change_oer.median():,.0f}
    Non-OER: ${income_change_non.median():,.0f}
    Difference: ${income_change_oer.median() - income_change_non.median():,.0f}

    AVERAGE SKILLS GAINED:
    OER: {oer_users['skills_final'].mean():.1f} points
    Non-OER: {non_oer_users['skills_final'].mean():.1f} points

    SKILLS THRESHOLD REACHED:
    ≥40 points: {(oer_users['skills_final'] >= 40).sum()}/{len(oer_users)} ({(oer_users['skills_final'] >= 40).mean()*100:.1f}%)
    ≥60 points: {(oer_users['skills_final'] >= 60).sum()}/{len(oer_users)} ({(oer_users['skills_final'] >= 60).mean()*100:.1f}%)
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('oer_simulation_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'oer_simulation_results.png'")
        plt.show()

if __name__ == "__main__":
    sim = OERSimulation(n_agents=10000, n_quarters=40, random_seed=42)
    results_df = sim.run_simulation()
    merged_results = sim.analyze_results(results_df)
    sim.create_visualizations(results_df, merged_results)
    
    results_df.to_csv('simulation_results.csv', index=False)
    merged_results.to_csv('final_comparison.csv', index=False)
    print("\nData saved to 'simulation_results.csv' and 'final_comparison.csv'")
