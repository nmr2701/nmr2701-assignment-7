from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with your own secret key, need this for session management

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    X = np.random.rand(N)  # Generate random values for X

    # Y = beta0 + beta1 * X + mu + error term
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Generate Y

    # TODO 3: Fit a linear regression model to X and Y
    model = LinearRegression()
    # None  # Fit the model to X and Y
    model.fit(X.reshape(-1, 1), Y)  # Fit the model to X and Y
    slope = model.coef_[0]  # Extract the slope (coefficient) from the fitted model
    intercept = model.intercept_  # Extract the intercept from the fitted model

    # TODO 4: Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.scatter(X, Y, color='blue')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red')  # Fitted line
    plt.title("Scatter plot with regression line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(plot1_path)  # Save the scatter plot
    plt.close()


    # TODO 5: Run S simulations to generate slopes and intercepts under the null hypothesis
    slopes = []
    intercepts = []

    for _ in range(S):
        # TODO 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N)  # Generate simulated X values
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Generate simulated Y values

        # TODO 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()  # Fit the model
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)  # Fit model
        sim_slope = sim_model.coef_[0]  # Extract slope from sim_model
        sim_intercept = sim_model.intercept_  # Extract intercept from sim_model

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)


    # TODO 8: Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.hist(slopes, bins=30, alpha=0.5, color='blue', label='Slopes')
    plt.hist(intercepts, bins=30, alpha=0.5, color='red', label='Intercepts')
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot2_path)  # Save the histogram plot
    plt.close()

    # TODO 9: Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_more_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

    # Return data needed for further analysis
    return X, Y, slope, intercept, plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        X, Y, slope, intercept, plot1, plot2, slope_extreme, intercept_extreme = generate_data(
            N, mu, beta0, beta1, sigma2, S
        )

        # Store data in session
        session['X'] = X.tolist()
        session['Y'] = Y.tolist()
        session['slope'] = slope
        session['intercept'] = intercept
        session['slope_extreme'] = slope_extreme
        session['intercept_extreme'] = intercept_extreme
        session['N'] = N
        session['mu'] = mu
        session['sigma2'] = sigma2
        session['beta0'] = beta0
        session['beta1'] = beta1
        session['S'] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = session.get('N')
    mu = session.get('mu')
    sigma2 = session.get('sigma2')
    beta0 = session.get('beta0')
    beta1 = session.get('beta1')
    S = session.get('S')
    X = np.array(session.get('X'))
    Y = np.array(session.get('Y'))
    slope = session.get('slope')
    intercept = session.get('intercept')
    slope_extreme = session.get('slope_extreme')
    intercept_extreme = session.get('intercept_extreme')

    # Check if S is None
    if S is None:
        return "Error: Number of simulations (S) is not set in the session.", 400

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Set hypothesized_value based on the parameter from data generation
    if parameter == "slope":
        hypothesized_value = beta1
        observed_stat = slope
    elif parameter == "intercept":
        hypothesized_value = beta0
        observed_stat = intercept
    else:
        hypothesized_value = None
        observed_stat = None

    # TODO 10: Perform hypothesis testing based on the inputs
    # Use the data generation parameters and the selected parameter to test



    # TODO 11: Implement the hypothesis testing simulations
    # You will need to:
    # - Generate simulated datasets under the null hypothesis
    # - Fit models and collect the statistics
    # - Compare with the observed statistic
    # - Implement p-value calculations (students will implement this)

    # Initialize list to store simulated parameter estimates under null hypothesis
    simulated_stats = []

    for _ in range(S):
        # TODO 12: Generate simulated datasets under the null hypothesis
        X_sim = np.random.rand(N)  # Generate simulated X values
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Generate simulated Y values under null hypothesis

        # TODO 13: Fit linear regression to simulated data and extract parameter
        sim_model = LinearRegression()  # Fit model
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)  # Fit model
        sim_stat = sim_model.coef_[0]  # Extract parameter being tested (slope)

        simulated_stats.append(sim_stat)

    # TODO 14: Calculate p-value based on test type
    if test_type == "two-sided":
    # Two-sided test: p-value is twice the proportion of simulated stats more extreme
        if len(simulated_stats) > 0 and observed_stat is not None:
            simulated_stats = np.array(simulated_stats)  # Ensure it's a NumPy array
            p_value = 2 * np.mean(np.abs(simulated_stats) >= np.abs(observed_stat))
        else:
            p_value = None  # Handle case where stats are not valid
    elif test_type == "less":
        # One-sided test (less): p-value is the proportion of simulated stats less than or equal to observed stat
        if len(simulated_stats) > 0 and observed_stat is not None:
            simulated_stats = np.array(simulated_stats)  # Ensure it's a NumPy array
            p_value = np.mean(simulated_stats <= observed_stat)
        else:
            p_value = None  # Handle case where stats are not valid
    else:
        # One-sided test (greater): p-value is the proportion of simulated stats greater than or equal to observed stat
        if len(simulated_stats) > 0 and observed_stat is not None:
            simulated_stats = np.array(simulated_stats)  # Ensure it's a NumPy array
            p_value = np.mean(simulated_stats >= observed_stat)
        else:
            p_value = None  # Handle case where stats are not valid


    # TODO 15: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    fun_message = "Congratulations! You've discovered something amazing!" if p_value <= 0.0001 else None

    # TODO 16: Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    # Replace with code to generate and save the plot
    plt.hist(simulated_stats, bins=30, alpha=0.5, color='blue', label='Simulated Stats')
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.title("Histogram of Simulated Statistics")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot3_path)  # Save the histogram plot
    plt.close()


    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
        slope_extreme=slope_extreme,
        intercept_extreme=intercept_extreme,
        # TODO 17: Uncomment the following lines when implemented
        p_value=p_value,  # Students will include this when implemented
        fun_message=fun_message,  # Students will include this when implemented
    )



@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = session.get('N')
    mu = session.get('mu')
    sigma2 = session.get('sigma2')
    beta0 = session.get('beta0')
    beta1 = session.get('beta1')
    S = session.get('S')
    X = np.array(session.get('X'))
    Y = np.array(session.get('Y'))
    slope = session.get('slope')
    intercept = session.get('intercept')
    slope_extreme = session.get('slope_extreme')
    intercept_extreme = session.get('intercept_extreme')


    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))


    if confidence_level < 0 or confidence_level > 100:
        print("EEOEOEOEOEOEOOEOEO: Confidence level must be between 0 and 100.")
        return "Error: Confidence level must be between 0 and 100.", 400

    # Generate observed statistic
    observed_stat = slope if parameter == "slope" else intercept

    # TODO 18: Perform bootstrap simulations to calculate confidence intervals
    # Use the data generated earlier and the selected parameter

    # Initialize list to store bootstrap parameter estimates
    bootstrap_stats = []

    for _ in range(S):
        # TODO 19: Generate bootstrap sample by resampling with replacement
        indices = np.random.choice(N, N, replace=True)  # Generate random indices with replacement
        X_bootstrap = X[indices]  # Resample X based on indices
        Y_bootstrap = Y[indices]  # Resample Y based on indices

        # TODO 20: Fit linear regression to bootstrap sample and extract parameter
        bootstrap_model = LinearRegression().fit(X_bootstrap.reshape(-1, 1), Y_bootstrap)  # Fit model
        boot_stat = bootstrap_model.coef_[0]
        bootstrap_stats.append(boot_stat)


    mean_estimate = np.mean(bootstrap_stats)  # Calculate mean of bootstrap estimates


    # TODO 21: Calculate confidence interval based on bootstrap_stats and confidence_level
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    conf_interval = np.percentile(bootstrap_stats, [lower_percentile, upper_percentile])
    print("cond_interval", conf_interval)

     # Extract lower and upper confidence interval values
    ci_lower = conf_interval[0]  # Lower bound of the confidence interval
    ci_upper = conf_interval[1]  # Upper bound of the confidence interval


    # TODO 22: Plot bootstrap distribution with confidence interval
    plot4_path = "static/plot4.png"
    plt.hist(bootstrap_stats, bins=30, alpha=0.5, color='blue', label='Bootstrap Stats')
    plt.axvline(conf_interval[0], color='red', linestyle='dashed', linewidth=2, label='Lower CI')
    plt.axvline(conf_interval[1], color='red', linestyle='dashed', linewidth=2, label='Upper CI')
    plt.axvline(observed_stat, color='green', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.title("Bootstrap Distribution with Confidence Interval")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()



    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        conf_interval=conf_interval,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        mean_estimate=mean_estimate,  # Pass mean_estimate to the template
        S=S,
        slope_extreme=slope_extreme,
        intercept_extreme=intercept_extreme,
    )


if __name__ == "__main__":
    app.run(debug=True)