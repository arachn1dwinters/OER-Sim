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
    def __init__(self, n_agents=10000, n_quarters=40, random_seed=42):
        np.random.seed(random_seed)
        self.n_agents = n_agents
        self.n_quarters = n_quarters
        self.agents: List[Agent] = []
        self.results = []
        
        # Ranges for income quintiles (Q1, Q2, Q3, Q4, Q5)
        self.income_quintiles = [
            (0, 29963),
            (29964, 59533),
            (59534, 96562),
            (96563, 160832),
            (160833, 500000)
        ]
        
        # "pct" being the percent of people who occupy each education level or engagement level w/ OER
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
            'github': {'hours': 200, 'boost': 0.12},
            'projects': {'hours': 150, 'boost': 0.08},
            'opensource': {'hours': 250, 'boost': 0.15},
            'freelance': {'hours': 300, 'boost': 0.10},
            'certification': {'hours': 70, 'cost': 1000, 'boost': 0.18}
        }
        
    def initialize_agents(self):
        for i in range(self.n_agents):
            age = np.random.uniform(24, 45)
            household_size = max(1, np.random.normal(2.5, 1.3))
            
            quintile = np.random.choice([0, 1, 2, 3, 4], p=[0.20, 0.20, 0.20, 0.20, 0.20])
            income_min, income_max = self.income_quintiles[quintile]
            income = np.random.uniform(income_min, income_max)
            
            employment = np.random.choice(['employed', 'unemployed', 'not_in_lf'], 
                                         p=[0.95, 0.03, 0.02])
            
            if quintile == 0:
                wealth = np.random.uniform(0, 1000)
            else:
                wealth = np.random.exponential(scale=5000 * (quintile + 1))
            
            internet_probs = [0.78, 0.85, 0.90, 0.95, 0.99]
            internet_access = np.random.random() < internet_probs[quintile]
            
            education_choice = np.random.choice(
                list(self.education_levels.keys()),
                p=[0.08, 0.27, 0.29, 0.24, 0.12]
            )
            education_multiplier = self.education_levels[education_choice]['multiplier']
            income *= education_multiplier
            
            study_time_base = max(0, np.random.normal(10, 5))
            study_time = study_time_base * (1 - 0.4 * (household_size - 2.5) / 1.3)
            study_time = study_time * (1 + 0.2 * (quintile - 2) / 2)
            study_time = max(0, study_time)
            
            learning_capacity = np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3)
            
            oer_aware = np.random.random() < 0.35
            oer_engaged = False
            if oer_aware and internet_access:
                oer_engaged = np.random.random() < 0.15
            
            engagement_level = 'none'
            if oer_engaged:
                engagement_level = np.random.choice(
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
    
    def acquire_skills(self, agent: Agent):
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
        
        skill_gain = (actual_hours / 25) * agent.learning_capacity * efficiency * material_quality
        agent.skills += skill_gain
        agent.quarters_engaged += 1
    
    def select_signal(self, agent: Agent):
        if agent.skills < 40 or agent.signal_completed:
            return
        
        time_available_total = agent.study_time_available * 13
        
        # Certifications cost money, so there is a wealth threshold necessary for the cert. choice
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
    
    def complete_signal(self, agent: Agent):
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
    
    def update_income(self, agent: Agent):
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
        
        random_shock = np.random.normal(1.0, 0.10)
        
        income_multiplier = (1 + growth_rate + skill_premium + transition_boost - credential_discount)
        agent.income = agent.income * income_multiplier * random_shock
        agent.income = max(0, agent.income)
    
    def update_employment(self, agent: Agent):
        job_change_prob = self.calculate_job_change_probability(agent)
        
        if np.random.random() < job_change_prob:
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
                self.acquire_skills(agent)
                self.select_signal(agent)
                self.complete_signal(agent)
                self.update_employment(agent)
                self.update_income(agent)
                
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
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        quintile_transitions = np.zeros((5, 5))
        for _, row in merged.iterrows():
            quintile_transitions[int(row['quintile_initial']), int(row['quintile_final'])] += 1
        
        for i in range(5):
            if quintile_transitions[i].sum() > 0:
                quintile_transitions[i] = quintile_transitions[i] / quintile_transitions[i].sum()
        
        sns.heatmap(quintile_transitions, annot=True, fmt='.2f', cmap='YlGnBu', 
                   xticklabels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                   yticklabels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Quintile Transition Matrix')
        axes[0, 0].set_xlabel('Final Quintile')
        axes[0, 0].set_ylabel('Initial Quintile')
        
        oer_mobility = merged.groupby('oer_engaged_initial').apply(
            lambda x: (x['quintile_final'] > x['quintile_initial']).mean()
        )
        axes[0, 1].bar(['Non-OER Users', 'OER Users'], oer_mobility.values, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 1].set_title('Upward Mobility Rate by OER Usage')
        axes[0, 1].set_ylabel('Proportion with Upward Mobility')
        axes[0, 1].set_ylim([0, 1])
        
        income_over_time_oer = df[df['agent_id'].isin(
            df[df['quarter'] == 0][df['oer_engaged'] == True]['agent_id']
        )].groupby('quarter')['income'].mean()
        
        income_over_time_non = df[df['agent_id'].isin(
            df[df['quarter'] == 0][df['oer_engaged'] == False]['agent_id']
        )].groupby('quarter')['income'].mean()
        
        axes[0, 2].plot(income_over_time_oer.index, income_over_time_oer.values, 
                       label='OER Users', linewidth=2, color='#4ECDC4')
        axes[0, 2].plot(income_over_time_non.index, income_over_time_non.values, 
                       label='Non-OER Users', linewidth=2, color='#FF6B6B')
        axes[0, 2].set_title('Average Income Over Time')
        axes[0, 2].set_xlabel('Quarter')
        axes[0, 2].set_ylabel('Average Income ($)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        skills_dist = merged[merged['oer_engaged_initial'] == True]['skills_final']
        axes[1, 0].hist(skills_dist, bins=30, edgecolor='black', color='#95E1D3')
        axes[1, 0].set_title('Skills Distribution (OER Users)')
        axes[1, 0].set_xlabel('Skill Points')
        axes[1, 0].set_ylabel('Number of Agents')
        axes[1, 0].axvline(40, color='red', linestyle='--', label='Threshold (40)')
        axes[1, 0].axvline(60, color='orange', linestyle='--', label='Signal Eligible (60)')
        axes[1, 0].legend()
        
        by_quintile = merged.groupby('quintile_initial').agg({
            'quintile_final': lambda x: (x > merged.loc[x.index, 'quintile_initial']).mean()
        })
        axes[1, 1].bar(range(1, 6), by_quintile['quintile_final'].values, color='#F38181')
        axes[1, 1].set_title('Upward Mobility by Starting Quintile')
        axes[1, 1].set_xlabel('Starting Quintile')
        axes[1, 1].set_ylabel('Proportion with Upward Mobility')
        axes[1, 1].set_xticks(range(1, 6))
        axes[1, 1].set_ylim([0, 1])
        
        income_change_oer = merged[merged['oer_engaged_initial'] == True]['income_final'] - merged[merged['oer_engaged_initial'] == True]['income_initial']
        income_change_non = merged[merged['oer_engaged_initial'] == False]['income_final'] - merged[merged['oer_engaged_initial'] == False]['income_initial']
        
        axes[1, 2].boxplot([income_change_non, income_change_oer], 
                          labels=['Non-OER Users', 'OER Users'])
        axes[1, 2].set_title('Income Change Distribution')
        axes[1, 2].set_ylabel('Income Change ($)')
        axes[1, 2].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
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
