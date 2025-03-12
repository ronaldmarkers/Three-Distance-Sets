#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <tuple>
#include <numeric>
#include <iostream>
#include <memory>

#include "ortools/linear_solver/linear_solver.h"

using namespace std;
using namespace operations_research;

// Binomial Coefficient Program: https://stackoverflow.com/questions/55421835/c-binomial-coefficient-is-too-slow
int binomial(const int n, const int k) {
    std::vector<int> aSolutions(k);
    if (k == 0) return 1;  // Base case: binomial(n, 0) = 1
    if (k > n) return 0;  // Binomial coefficient is 0 when k > n

    aSolutions[0] = n - k + 1;

    for (int i = 1; i < k; ++i) {
        aSolutions[i] = aSolutions[i - 1] * (n - k + 1 + i) / (i + 1);
    }

    return aSolutions[k - 1];
}

vector<int> error = { 23 };
vector<int> hold;
vector<vector<int>> fail;
vector<double> values;
int q = 2;

int krav(int k, int i, int n, int q) {
    int sum = 0;
    for (int j = 0; j <= k; j++) {
        sum += pow(-q, j) * pow(q - 1, k - j) * binomial(n - j, k - j) * binomial(i, j);
    }
    return sum;
}

double solve_optimization(int n, const vector<int>& dis, int q) {
    // Create solver
    unique_ptr<MPSolver> solver(MPSolver::CreateSolver("GLOP"));
    if (!solver) {
        cout << "GLOP solver unavailable." << endl;
        return -1;
    }

    // Define variables
    MPVariable* x1 = solver->MakeNumVar(0, solver->infinity(), "x1");
    MPVariable* x2 = solver->MakeNumVar(0, solver->infinity(), "x2");
    MPVariable* x3 = solver->MakeNumVar(0, solver->infinity(), "x3");

    // Objective function: maximize x1 + x2 + x3 + 1
    MPObjective* objective = solver->MutableObjective();
    objective->SetCoefficient(x1, 1);
    objective->SetCoefficient(x2, 1);
    objective->SetCoefficient(x3, 1);
    objective->SetOffset(1);
    objective->SetMaximization();

    // Constraints
    for (int p = 0; p <= n; p++) {
        MPConstraint* constraint = solver->MakeRowConstraint(0, solver->infinity());

        double tot = krav(p, 0, n, q);
        if (dis[0] <= n) tot += krav(p, dis[0], n, q) * x1->solution_value();
        if (dis[1] <= n) tot += krav(p, dis[1], n, q) * x2->solution_value();
        if (dis[2] <= n) tot += krav(p, dis[2], n, q) * x3->solution_value();

        constraint->SetCoefficient(x1, krav(p, dis[0], n, q));
        constraint->SetCoefficient(x2, krav(p, dis[1], n, q));
        constraint->SetCoefficient(x3, krav(p, dis[2], n, q));
        constraint->SetBounds(tot, solver->infinity());
    }

    // Solve the problem
    auto result_status = solver->Solve();
    if (result_status != MPSolver::OPTIMAL && result_status != MPSolver::FEASIBLE) {
        cout << "No solution found for n = " << n << ", dis = ("
            << dis[0] << ", " << dis[1] << ", " << dis[2] << ")"
            << " with status code: " << result_status << endl;
        return -1;  // Return a special value indicating no solution
    }

    return objective->Value();
}

int main() {
    for (int n = 45; n <= 60; n++) {
        vector<vector<int>> index;
        int len = 0;
        vector<double> value;

        for (int i = 1; i <= n; i++) {
            for (int j = i + 1; j <= n; j++) {
                for (int k = j + 1; k <= n; k++) {
                    if ((i + j + k >= ((3 * n) / 2) + 1) || (-((n / 2) - i) * ((n / 2) - j) * ((n / 2) - k) + (n / 4) * (i + j + k) - (3 * pow(n, 2) / 8) > 0)) {
                        int Coe = 1 + n + (n * (n - 1)) / 2;
                        int U = floor(0.5 + sqrt(pow(Coe, 2) / (2 * Coe - 2) + 0.25));

                        if (n + (n * (n - 1) * (n - 2)) / 6 >= 2 * U) {
                            if (floor((double)j / (j - i) * k / (k - i)) == (double)j / (j - i) * k / (k - i)) {
                                if (floor((double)i / (i - j) * k / (k - j)) == (double)i / (i - j) * k / (k - j)) {
                                    if (floor((double)i / (i - k) * j / (j - k)) == (double)i / (i - k) * j / (j - k)) {
                                        if ((double)j / (j - i) * k / (k - i) <= U) {
                                            if ((double)i / (j - i) * k / (k - j) <= U) {
                                                if ((double)i / (k - i) * j / (k - j) <= U) {
                                                    index.push_back({ i, j, k });
                                                    len++;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            index.push_back({ i, j, k });
                            len++;
                        }
                    }
                }
            }
        }

        for (int ind = 0; ind < len; ind++) {
            vector<int> dis = index[ind];
            double A = solve_optimization(n, dis, q);

            value.push_back(A);
            if (A >= n + (n * (n - 1) * (n - 2)) / 6) {
                fail.push_back({ n, (int)A, dis[0], dis[1], dis[2] });
            }
        }

        double G = *max_element(value.begin(), value.end());
        values.push_back(G);
        if (G <= n + (n * (n - 1) * (n - 2)) / 6) {
            hold.push_back(n);
        }
    }

    cout << "Hold: ";
    for (int h : hold) cout << h << " ";
    cout << endl;

    cout << "Fail: " << endl;
    for (const auto& f : fail) {
        cout << "{";
        for (size_t i = 0; i < f.size(); i++) {
            cout << f[i];
            if (i < f.size() - 1) cout << ", ";
        }
        cout << "}, ";
    }
    cout << endl;

    return 0;
}


// Binomial Coefficient Program: https://stackoverflow.com/questions/55421835/c-binomial-coefficient-is-too-slow
int binomial(const int n, const int k) {
    std::vector<int> aSolutions(k);
    if (k == 0) return 1;  // Base case: binomial(n, 0) = 1
    if (k > n) return 0;  // Binomial coefficient is 0 when k > n

    aSolutions[0] = n - k + 1;

    for (int i = 1; i < k; ++i) {
        aSolutions[i] = aSolutions[i - 1] * (n - k + 1 + i) / (i + 1);
    }

    return aSolutions[k - 1];
}