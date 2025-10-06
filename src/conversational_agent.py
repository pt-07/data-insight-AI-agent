import anthropic
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class ConversationalAnalysisAgent:
    def __init__(self, datasets):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.datasets = datasets
        self.conversation_history = []
        self.tools = self._define_tools()
        
    def _define_tools(self):
        """Define tools the agent can use to query data"""
        return [
            {
                "name": "get_user_orders",
                "description": "Get order history and shopping patterns for a specific user ID",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "integer",
                            "description": "The user ID to query"
                        }
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "analyze_product",
                "description": "Get statistics and insights about a specific product or product name pattern",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "description": "Product name or keyword to search for"
                        }
                    },
                    "required": ["product_name"]
                }
            },
            {
                "name": "get_top_products",
                "description": "Get the most frequently ordered products, optionally filtered by department",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "department": {
                            "type": "string",
                            "description": "Department name to filter by (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of top products to return (default 10)"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "analyze_department",
                "description": "Get statistics about a department's performance and popular products",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "department_name": {
                            "type": "string",
                            "description": "Department name to analyze"
                        }
                    },
                    "required": ["department_name"]
                }
            },
            {
                "name": "find_product_pairs",
                "description": "Find products frequently bought together (market basket analysis)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "description": "Product to find common pairs for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of pairs to return (default 5)"
                        }
                    },
                    "required": ["product_name"]
                }
            },
            {
                "name": "get_reorder_stats",
                "description": "Get reorder statistics - which products have highest/lowest reorder rates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sort_by": {
                            "type": "string",
                            "description": "Sort by 'highest' or 'lowest' reorder rate",
                            "enum": ["highest", "lowest"]
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of products to return (default 10)"
                        }
                    },
                    "required": ["sort_by"]
                }
            },
            {
                "name": "create_visualization",
                "description": "Create a chart or graph to visualize data. Returns the filepath of the saved image.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "description": "Type of chart to create",
                            "enum": ["bar", "horizontal_bar", "line", "pie", "scatter"]
                        },
                        "data_source": {
                            "type": "string",
                            "description": "What data to visualize (e.g., 'top_products', 'department_comparison', 'reorder_rates')"
                        },
                        "title": {
                            "type": "string",
                            "description": "Title for the chart"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of items to include in visualization (default 10)"
                        },
                        "department_filter": {
                            "type": "string",
                            "description": "Optional department filter for data"
                        }
                    },
                    "required": ["chart_type", "data_source", "title"]
                }
            }
        ]
    
    def _execute_tool(self, tool_name, tool_input):
        """Execute the requested tool and return results"""
        
        if tool_name == "get_user_orders":
            return self._get_user_orders(tool_input["user_id"])
        
        elif tool_name == "analyze_product":
            return self._analyze_product(tool_input["product_name"])
        
        elif tool_name == "get_top_products":
            department = tool_input.get("department")
            limit = tool_input.get("limit", 10)
            return self._get_top_products(department, limit)
        
        elif tool_name == "analyze_department":
            return self._analyze_department(tool_input["department_name"])

        elif tool_name == "find_product_pairs":
            product = tool_input["product_name"]
            limit = tool_input.get("limit", 5)
            return self._find_product_pairs(product, limit)
        
        elif tool_name == "get_reorder_stats":
            sort_by = tool_input["sort_by"]
            limit = tool_input.get("limit", 10)
            return self._get_reorder_stats(sort_by, limit)
        
        elif tool_name == "create_visualization":
            return self._create_visualization(
                tool_input["chart_type"],
                tool_input["data_source"],
                tool_input["title"],
                tool_input.get("limit", 10),
                tool_input.get("department_filter")
            )
        return {"error": f"Unknown tool: {tool_name}"}
    
    def _get_user_orders(self, user_id):
        """Get detailed order information for a user"""
        orders = self.datasets['orders']
        order_products = self.datasets.get('order_products__train', self.datasets.get('order_products'))
        products = self.datasets['products']
        user_orders = orders[orders['user_id'] == user_id]
        if len(user_orders) == 0:
            return {"error": f"No orders found for user {user_id}"}
        
        order_ids = user_orders['order_id'].tolist()
        user_items = order_products[order_products['order_id'].isin(order_ids)]
        user_items = user_items.merge(products, on='product_id', how='left')
        top_products = user_items['product_name'].value_counts().head(10).to_dict()
        return {
            "user_id": int(user_id),
            "total_orders": int(len(user_orders)),
            "total_items": int(len(user_items)),
            "avg_cart_size": float(user_items.groupby('order_id').size().mean()),
            "top_products": {str(k): int(v) for k, v in top_products.items()}
        }
    
    def _analyze_product(self, product_name):
        """Analyze a specific product"""
        products = self.datasets['products']
        order_products = self.datasets.get('order_products__train', self.datasets.get('order_products'))
        matching = products[products['product_name'].str.contains(product_name, case=False, na=False)]
        
        if len(matching) == 0:
            return {"error": f"No products found matching '{product_name}'"}
        product_ids = matching['product_id'].tolist()
        product_orders = order_products[order_products['product_id'].isin(product_ids)]
        result = {
            "matching_products": matching['product_name'].tolist()[:10],
            "total_orders": int(len(product_orders)),
            "unique_customers": int(product_orders['order_id'].nunique()) if 'order_id' in product_orders.columns else 0
        }
        if 'reordered' in product_orders.columns:
            result["reorder_rate"] = float(product_orders['reordered'].mean() * 100)
        return result
    
    def _get_top_products(self, department=None, limit=10):
        """Get top products overall or by department"""
        products = self.datasets['products']
        order_products = self.datasets.get('order_products__train', self.datasets.get('order_products'))
        merged = order_products.merge(products, on='product_id', how='left')
        #filter by department if specified
        if department and 'departments' in self.datasets:
            departments = self.datasets['departments']
            dept_ids = departments[departments['department'].str.contains(department, case=False, na=False)]['department_id'].tolist()
            merged = merged[merged['department_id'].isin(dept_ids)]
        top_products = merged['product_name'].value_counts().head(limit)
        
        return {
            "department_filter": department,
            "top_products": {str(k): int(v) for k, v in top_products.items()}
        }
    
    def _analyze_department(self, department_name):
        """Analyze a specific department"""
        if 'departments' not in self.datasets:
            return {"error": "Departments data not available"}
        
        departments = self.datasets['departments']
        products = self.datasets['products']
        order_products = self.datasets.get('order_products__train', self.datasets.get('order_products'))
        matching_dept = departments[departments['department'].str.contains(department_name, case=False, na=False)]
        if len(matching_dept) == 0:
            return {"error": f"No department found matching '{department_name}'"}
        dept_id = matching_dept.iloc[0]['department_id']
        dept_products = products[products['department_id'] == dept_id]
        #orders for this department
        dept_orders = order_products[order_products['product_id'].isin(dept_products['product_id'])]
        return {
            "department_name": matching_dept.iloc[0]['department'],
            "total_products": int(len(dept_products)),
            "total_orders": int(len(dept_orders)),
            "unique_products_ordered": int(dept_orders['product_id'].nunique())
        }
    
    def _find_product_pairs(self, product_name, limit=5):
        """Find products commonly bought with the specified product"""
        products = self.datasets['products']
        order_products = self.datasets.get('order_products__train', self.datasets.get('order_products'))
        #find product
        matching = products[products['product_name'].str.contains(product_name, case=False, na=False)]
        if len(matching) == 0:
            return {"error": f"No products found matching '{product_name}'"}
        product_id = matching.iloc[0]['product_id']
        orders_with_product = order_products[order_products['product_id'] == product_id]['order_id'].unique
        other_products = order_products[
            (order_products['order_id'].isin(orders_with_product)) & 
            (order_products['product_id'] != product_id)
        ]
        #count co-occurrences
        paired = other_products.merge(products, on='product_id', how='left')
        top_pairs = paired['product_name'].value_counts().head(limit)
        return {
            "product": matching.iloc[0]['product_name'],
            "commonly_bought_with": {str(k): int(v) for k, v in top_pairs.items()}
        }
    
    def _get_reorder_stats(self, sort_by, limit=10):
        """Get products with highest/lowest reorder rates"""
        products = self.datasets['products']
        order_products = self.datasets.get('order_products__train', self.datasets.get('order_products'))
        if 'reordered' not in order_products.columns:
            return {"error": "Reorder data not available"}
        #reorder rate by product
        reorder_rates = order_products.groupby('product_id')['reordered'].agg(['mean', 'count'])
        reorder_rates = reorder_rates[reorder_rates['count'] >= 10]  # Min 10 orders
        if sort_by == "highest":
            reorder_rates = reorder_rates.sort_values('mean', ascending=False)
        else:
            reorder_rates = reorder_rates.sort_values('mean', ascending=True)
        top_products = reorder_rates.head(limit)
        top_products = top_products.merge(products[['product_id', 'product_name']], on='product_id', how='left')
        result = {}
        for _, row in top_products.iterrows():
            result[row['product_name']] = {
                "reorder_rate": float(row['mean'] * 100),
                "total_orders": int(row['count'])
            }
        return {
            "sort_by": sort_by,
            "products": result
        }
    
    def _create_visualization(self, chart_type, data_source, title, limit=10, department_filter=None):
        """Create a visualization and save to file"""
        
        # Create visualizations directory if it doesn't exist
        os.makedirs("visualizations", exist_ok=True)
        
        # Generate data based on source
        if data_source == "top_products":
            data = self._get_top_products(department_filter, limit)
            if "error" in data:
                return data
            plot_data = data["top_products"]
            
        elif data_source == "reorder_rates":
            data = self._get_reorder_stats("highest", limit)
            if "error" in data:
                return data
            plot_data = {k: v["reorder_rate"] for k, v in data["products"].items()}
            
        elif data_source == "department_comparison":
            if 'departments' not in self.datasets:
                return {"error": "Department data not available"}
            
            departments = self.datasets['departments']
            products = self.datasets['products']
            order_products = self.datasets.get('order_products__train', self.datasets.get('order_products'))
            
            # Get order counts by department
            merged = order_products.merge(products, on='product_id', how='left')
            merged = merged.merge(departments, on='department_id', how='left')
            dept_counts = merged['department'].value_counts().head(limit)
            plot_data = dept_counts.to_dict()
            
        else:
            return {"error": f"Unknown data source: {data_source}"}
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        if chart_type == "bar":
            plt.bar(range(len(plot_data)), list(plot_data.values()), color='steelblue')
            plt.xticks(range(len(plot_data)), list(plot_data.keys()), rotation=45, ha='right')
            plt.ylabel('Count')
            
        elif chart_type == "horizontal_bar":
            plt.barh(range(len(plot_data)), list(plot_data.values()), color='steelblue')
            plt.yticks(range(len(plot_data)), list(plot_data.keys()))
            plt.xlabel('Count')
            
        elif chart_type == "pie":
            plt.pie(list(plot_data.values()), labels=list(plot_data.keys()), autopct='%1.1f%%')
            
        elif chart_type == "line":
            plt.plot(range(len(plot_data)), list(plot_data.values()), marker='o', linewidth=2, markersize=8)
            plt.xticks(range(len(plot_data)), list(plot_data.keys()), rotation=45, ha='right')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            
        elif chart_type == "scatter":
            plt.scatter(range(len(plot_data)), list(plot_data.values()), s=100, alpha=0.6, color='steelblue')
            plt.xticks(range(len(plot_data)), list(plot_data.keys()), rotation=45, ha='right')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualizations/{data_source}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "success": True,
            "filepath": filename,
            "message": f"Visualization saved to {filename}",
            "data_points": len(plot_data)
        }
    
    def ask(self, question):
        """Ask a question about the data"""
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        
        print(f"\n🤔 Thinking...")
        
        # Call Claude with tools
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=self.tools,
            messages=self.conversation_history
        )
        
        # Process response and handle tool calls
        while response.stop_reason == "tool_use":
            # Build assistant message content with all blocks
            assistant_content = []
            tool_uses = []
            
            for block in response.content:
                if block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
                    tool_uses.append(block)
                elif hasattr(block, "text"):
                    assistant_content.append({
                        "type": "text",
                        "text": block.text
                    })
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            # Execute all tools and collect results
            tool_results = []
            for tool_use_block in tool_uses:
                tool_name = tool_use_block.name
                tool_input = tool_use_block.input
                
                print(f"🔧 Using tool: {tool_name}")
                
                # Execute tool
                tool_result = self._execute_tool(tool_name, tool_input)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": json.dumps(tool_result)
                })
            
            # Add tool results to history
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })
            
            # Continue conversation with tool results
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=self.tools,
                messages=self.conversation_history
            )
        
        # Get final text response
        final_response = ""
        assistant_content = []
        
        for block in response.content:
            if hasattr(block, "text"):
                final_response += block.text
                assistant_content.append({
                    "type": "text",
                    "text": block.text
                })
        
        # Add final assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_content
        })
        
        return final_response if final_response else "No response generated"

def start_conversation(datasets):
    """Start interactive conversation mode"""
    agent = ConversationalAnalysisAgent(datasets)
    
    print("\n" + "="*60)
    print("CONVERSATIONAL DATA ANALYSIS AGENT")
    print("="*60)
    print("\nAsk me anything about your Instacart data!")
    print("Examples:")
    print("  - What are the top 10 most ordered products?")
    print("  - Which products are frequently bought together with bananas?")
    print("  - What's the reorder rate for organic products?")
    print("  - Analyze user 12345's shopping behavior")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            answer = agent.ask(question)
            print(f"\n🤖 Agent: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

# Example usage
if __name__ == "__main__":
    print("Import this module and call start_conversation(datasets)")