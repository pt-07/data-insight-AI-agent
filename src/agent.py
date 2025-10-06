import anthropic
import os
import json
import random
from dotenv import load_dotenv

load_dotenv()

class PersonaAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def prepare_user_data(self, datasets, num_users=5):
        """
        Prepare randomized user shopping data for persona analysis
        datasets: dict with 'orders', 'order_products', 'products', etc.
        """
        orders = datasets.get('orders')
        if 'order_products__train' in datasets:
            order_products = datasets['order_products__train']
        elif 'order_products' in datasets:
            order_products = datasets['order_products']
        else:
            raise KeyError("Could not find order_products dataset")
        products = datasets.get('products')
        user_ids = random.sample(list(orders['user_id'].unique()), num_users)
        user_profiles = []
        for user_id in user_ids:
            user_orders = orders[orders['user_id'] == user_id]
            order_ids = user_orders['order_id'].tolist()
            user_order_products = order_products[order_products['order_id'].isin(order_ids)]
            user_items = user_order_products.merge(
                products[['product_id', 'product_name', 'aisle_id', 'department_id']], 
                on='product_id', 
                how='left'
            )
            #shopping metrics
            total_orders = len(user_orders)
            total_items = len(user_items)
            avg_cart_size = user_items.groupby('order_id').size().mean()
            reorder_rate = user_order_products['reordered'].mean() if 'reordered' in user_order_products.columns else 0
            #top products and categories
            top_products = user_items['product_name'].value_counts().head(10).to_dict()
            if 'department_id' in user_items.columns and 'departments' in datasets:
                dept_counts = user_items.merge(
                    datasets['departments'], 
                    on='department_id'
                )['department'].value_counts().head(5).to_dict()
            else:
                dept_counts = {}
            # patterns
            if 'order_dow' in user_orders.columns:
                preferred_day = user_orders['order_dow'].mode()[0] if len(user_orders['order_dow'].mode()) > 0 else None
            else:
                preferred_day = None  
            if 'order_hour_of_day' in user_orders.columns:
                preferred_hour = user_orders['order_hour_of_day'].mode()[0] if len(user_orders['order_hour_of_day'].mode()) > 0 else None
            else:
                preferred_hour = None
            
            user_profiles.append({
                "user_id": int(user_id),
                "metrics": {
                    "total_orders": int(total_orders),
                    "total_items_purchased": int(total_items),
                    "avg_cart_size": round(float(avg_cart_size), 2),
                    "reorder_rate": round(float(reorder_rate * 100), 2)
                },
                "top_products": {str(k): int(v) for k, v in list(top_products.items())[:10]},
                "department_preferences": {str(k): int(v) for k, v in list(dept_counts.items())[:5]},
                "shopping_patterns": {
                    "preferred_day_of_week": int(preferred_day) if preferred_day is not None else None,
                    "preferred_hour": int(preferred_hour) if preferred_hour is not None else None
                }
            })
        return user_profiles
    
    def generate_personas(self, user_profiles):
        """
        Generate user personas using Claude AI
        """
        #  context for AI
        context = self._format_user_data(user_profiles)
        prompt = f"""You are a data analyst specializing in customer behavior and market segmentation. 
        
Analyze the shopping data for these {len(user_profiles)} users and create detailed user personas.

{context}

For EACH user, provide:
1. **Persona Name**: A creative, descriptive name (e.g., "Budget-Conscious Health Enthusiast")
2. **Demographic Profile**: Inferred age range, lifestyle, household type
3. **Shopping Behavior**: Key patterns and preferences
4. **Product Preferences**: What they buy and why
5. **Business Insights**: How to target/retain this customer
6. **Recommended Actions**: Specific marketing or product recommendations

Format each persona clearly with the user_id, then the analysis."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _format_user_data(self, user_profiles):
        """Format user data for AI consumption"""
        formatted= []
        for profile in user_profiles:
            user_str = f"""
USER ID: {profile['user_id']}
---
Metrics:
- Total Orders: {profile['metrics']['total_orders']}
- Total Items: {profile['metrics']['total_items_purchased']}
- Average Cart Size: {profile['metrics']['avg_cart_size']}
- Reorder Rate: {profile['metrics']['reorder_rate']}%

Top Products Purchased:
{self._format_dict(profile['top_products'])}

Department Preferences:
{self._format_dict(profile['department_preferences'])}

Shopping Patterns:
- Preferred Day: {self._day_name(profile['shopping_patterns']['preferred_day_of_week'])}
- Preferred Hour: {profile['shopping_patterns']['preferred_hour']}:00
"""
            formatted.append(user_str)
        
        return "\n".join(formatted)
    
    def _format_dict(self, d):
        """Pretty print dictionary"""
        return "\n".join([f"  • {k}: {v}" for k, v in d.items()])
    
    def _day_name(self, day_num):
        """Convert day number to name"""
        if day_num is None:
            return "N/A"
        days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return days[day_num] if 0 <= day_num < 7 else "N/A"

def generate_user_personas(datasets, num_users=5):
    """
    Main function to generate random user personas
    """
    agent = PersonaAgent()
    #prepare random user data
    print(f"Selecting {num_users} random users...")
    user_profiles = agent.prepare_user_data(datasets, num_users)
    print(f"Generating personas for users: {[p['user_id'] for p in user_profiles]}")
    #generate personas with agent
    personas = agent.generate_personas(user_profiles)
    return personas, user_profiles

if __name__ == "__main__":
    # Called from main.py
    print("Agent module ready. Import and use generate_user_personas(datasets)")