// /v1/recipe/{dish}
export type Dish = {
	dish: string;
	recipes: Recipe[];
};

export type Recipe = {
	id: number;
	name: string;
	ingredients: string[];
	instructions: string;
	source: string;
	servings: number;
};

// /v1/classify
export type ClassificationItem = {
	dish: string;
	confidence: number;
};

export interface Classify extends ClassificationItem {
	uncertain: boolean;
	top5: ClassificationItem[];
}
