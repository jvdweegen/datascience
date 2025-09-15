```python3
1.	Clean data
	- verwijder onnodige/overbodige kolommen
	- bepaal o.a. shape en unieke waardes van belangrijke kolommen (df.shape, df.info, df.value_counts)

	- overview maken	import ydata_profiling
				profile = ydata_profiling.ProfileReport(dt)
				profile

	
	- label encoding voor class
				preprocessing.LabelEncoder()
```
```python3
2. 	Visualize
	- melten en plotten: 	p = pd.DataFrame(X, columns = labels)
				p['class'] = y
				
				# prepare for plotting
				p = p.melt(id_vars='class')
				
				plt.figure(figsize=(10,5))
				sns.boxplot(data = p, x = 'variable', y='value', hue='class')

	- Facetgrid: 		g = sns.FacetGrid(p, col='class', row='variable', height=4)
				var = np.unique(p.variable)
				g.map(sns.histplot, 'value', kde=True)
```
```python3
3.	ML1
	- bepaal soort ML (KNeighbors, RandomForestClassifier, DummyClassifier, LogisticRegression, DecisionTreeClassifier)

	- splitten:		from sklearn.model_selection import train_test_split
				X = df.drop(['class'], axis=1)
				y = df['class']
				le = preprocessing.LaberEncoder()
				y_label = le.fit(y)
				y_transform = y_label.transform(y)
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.x, randomstate=...)

				eventueel nogmaals voor train set:
				X_train, X_validation, y_train, y_validation = train_test_split(X_trainval, y_trainval, random_state=1)

	- scalen:		from sklearn.preprocessing import StandardScaler
				scaler = StandardScaler()
				y = df['class'].values
				data = df.drop('class', axis=1)
				labels = data.columns
				X = scaler.fit_transform(data)

	- fitten		voorbeeld:
				'classifier' = neighbors.KNeighborsClassifier()
				'classifier'.fit(X_train, y_train)

	- evalueren/testen	'classifier'.score(X_test, y_test)

				from sklearn.metrics import confusion_matrix
				yhat = 'classifier'.predict(X_test) 
				ytrue = y_test
				labels = ['class']
				cfm = confusion_matrix(y_test, yhat, labels=labels)
				cfm

				clf_'classifier' = 'classifier'.fit(X_train, y_train)
				print('Accuracy of the Decision Tree Classifier on train set: {:.2f}'.format(clf_'classifier'.score(X_train, y_train)))
				print('Accuracy of the Decision Tree Classifier on test set: {:.2f}'.format(clf_'classifier'.score(X_test, y_test)))

				from sklearn.metrics import classification_report, accuracy_score
				print(accuracy_score(y_test, predictions))
				report = classification_report(y_test, predictions, target_names = iris.target_names)
				print(report)


	- cross validation	cross_val_score('classifier', X, y, cv=5)

				kfold = model_selection.KFold(n_splits=10)
				'classifier'CrossResults = model_selection.cross_val_score('classifier', X, y_transform, cv=kfold)
				'classifier2'CrossResults = model_selection.cross_val_score('classifier2', X, y_transform, cv=kfold)

				sns.boxplot(data=['classifier'CrossResults,
				                  'classifier2'CrossResults])
				plt.xticks([0, 1], [''classifier'', ''classifier2''])
				plt.xlabel('Model')
				plt.ylabel('Accuracy')
				plt.title('Cross validated accuracies')
				plt.show()
```
```python3
4.	GridSearch
	- bepalen parameters voor model (e.g. C (regularization parameter) of gamma (kernel bandwidth))
				_____________________________________________________________________________

				best_score = 0 
				for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
 					for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        					# Initialize SVC model for given combination of parameters
        					svm = SVC(gamma=gamma, C=C)
        					# Train on train set
        					svm.fit(X_train, y_train)
        					# Evaluate on validation set 
        					score = svm.score(X_validation, y_validation)
						# Store the best score 
                				if score > best_score: 
        						best_score = score 
        						best_parameters = {'C': C, 'gamma': gamma}
				_____________________________________________________________________________

				svm = SVC(**best_parameters)
				# Train on train/validation set
				svm.fit(X_trainval, y_trainval)
				# Score on test set
				test_score = svm.score(X_test, y_test)
				_____________________________________________________________________________

	- Gridsearch ook ingebouwd

				param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
				              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

				grid_search = GridSearchCV(SVC(), param_grid, cv=3) 

			cv = 2 aangeraden
				
				Voor beste parameters:
					grid_search.best_param_ 

				Voor beste score:
					grid_search.best_score_
	
				Voor Cross Validation resultaten:
					grid_search.cv_results_

	- visualizeren		results = pd.DataFrame(grid_search.cv_results_)
				# Reshape test scores and plot heatmap
				scores = np.array(results.mean_test_score).reshape(7, 6)
				ax = sns.heatmap(scores, xticklabels=param_grid['gamma'], yticklabels=param_grid['C'], annot=True)
				ax.set(xlabel='gamma', ylabel='C')
```
```python3
5.	Pipeline
	- sklearn object dat stappen volgt uit een toegevoegde [LIJST] met (TUPLES)
				Iedere tuple bevat gespecificeerde naam en instance van estimator.
				e.g. Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])

	- .fit(X_train, y_train) aanroepen op Pipeline voor trainen model
	  .score(X_test, y_test) aanroepen voor score

	- Pipeline kan in een GridSearch worden gedaan
				Wanneer je dit doet, moet je in de Dictionary met parameter grids (para_grid) aangeven bij welke stap in de Pipeline deze hoort
				Voorbeeld:
					param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              						'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
				Deze is dus voor de "svm" estimator

				GridSearchCV('pipeline', param_grid=param_grid, cv=2)


#.	Ensemble, Baggin, Boosting
	- ensemble		VotingClassifier(['lijst van estimators'])
	- bagging		BaggingClassifier(estimator='model', n_estimators='number_of_trees')
				RandomForestClassifier(n_estimators='number_of_trees', max_features='max_features')
				ExtraTreesClassifier(n_estimators='number_of_trees', max_features='max_features')
	- boosting		AdaBoostClassifier(n_estimators='number_of_trees')
				GradientBoostingClassifier(n_estimators='number_of_trees')
				
				XG boost
```
