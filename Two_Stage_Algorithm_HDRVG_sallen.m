%%%%%%Two Stage Vehicle Routing Problem Algorithm for HDRVG%%%%%%%%%^M

%%%Author: Stephanie Allen
% Last Updated: 5/9/2017 
% Content: Two Stage Vehicle Routing Problem Algorithm for Stephanie
% Allen's Honors Capstone Thesis

% Main sources: See paper

%%%Set Up: To run the code, make sure that our project folder is your
%%%'directory' in MATLAB.  

%%%Input: The only thing you need to input is the payload capacity of the
%%%vehicles and the number of iterations you would like the simulated
%%%annealing algorithm to go through for each "temperature."

%%%Description: This script runs through the two stage VRP algorithm 
%%%discussed in Stephanie Allen's Capstone Honors Thesis.  The first stage
%%%of the algorithm assigns nodes to vehicles using an integer program, and
%%%the second stage routes the vehicles through their assigned nodes using
%%%one of three heuristics/metaheuristics.  The
%%%script is calibrated to run the algorithm for each of the 26 days the
%%%HDRVG delivered supplies in Nepal.  The user can UNCOMMENT any of the 
%%%three different routing algorithms as needed to get output regarding the
%%%routing under each of these three algorithms.
%%%The algorithm can be re-calibrated for similar routing situations.

%%%Output: The script outputs a large amount of information (see the
%%%"Storing Output Data" section below).  The most important information it
%%%outputs is contained in the "all_v_node_orderings" variable, which is
%%%a cell array that contains 26 cell arrays - each of which has the
%%%routing information for a given day's vehicles (the ordering of the
%%%nodes for each vehicle).  The numbers correspond to the data points in
%%%the 26 CSV files.  Another key data structure is the
%%%"total_distance_each_day" variable which records the total distance
%%%traveled for a given day by all of the vehicles.


%%%%%%%%%%%Initial Set up Before Running the Algorithm%%%%%%%%%%%%%%%%%

clear 
close all

counter = 1; 
food_totals = csvread('Total_supplies_each_Day.csv');

%Tracking Vectors
assignment_probs = 0; %see if the sum of the decision variable values is different from number of nodes
convex_hull_probs = 0;

%Storing Output Data
all_assignments = cell(26,1);
raw_sol = cell(26,1);
all_distance_matrices = cell(26,1); %saves all the distance matrices
all_v_node_orderings = cell(26,1);
checking_routing = cell(26,1);
ck_routing_sorted = cell(26,1); %sorted data from checking routing
num_v = zeros(26,1); %number of vehicles for each day
num_each_node_all = zeros(26,1);
total_distance_each_route = cell(26,1); %distance for each route in each day
total_distance_each_day = zeros(26,1); %total distance traveled each day by all vehicles
number_nodes = zeros(26,1); %finding number of nodes for each day 

payload = input('Input the payload capacity ');
num_each_temp = input('Input the number of iterations at each temp for simulated annealing ');

tic
while counter < 27 

%%%%%Data Preparing%%%%%
clear prob_with_hull  %variables I want to clear at the beginning of each loop
prob_with_hull = 0;
clear corners_sifted
clear corners_sifted_1

file_name = ['Lat_Long_' num2str(counter) '_' '.csv'];        
data = csvread(file_name);

lat = data(:,1);
long = data(:,2);
[n,~] = size(data); %tells me the number of rows - number of nodes + 1 for yellow house (each data file has yellow
                    %house coordinates at the end)

number_nodes(counter) = (n-1);

distance_matrix = zeros(n,n); %will record the 
%distances between nodes i,j a(i,j) == a(j,i)


%%%%Creating the Distance matrix - includes distance to Yellow House (node n) %%%%%%%
for i = 1:n
   for j = i:n
       if i == j
           distance_matrix(i,i) = 0;
       else    
           p_1 = norm(data(i,:) - data(j,:)); %(LA textbook, pg 23)
           distance_matrix(i,j) = p_1;
           distance_matrix(j,i) = p_1;
       end
   end
end

all_distance_matrices{counter} = distance_matrix;

%%%%%Making the 'g' matrix for the Objective Function%%%%%%%%%%

%%%%Pre-work - Establishing the Seeds

%Finding the Convex Hull and then taking indices
if length(lat) > 2
    corners = convhull(lat, long);
    truth_corners = (corners == n);  %below - RECENTLY ADDED 4/11/2017
    if sum(truth_corners) > 0 %if any corners are equal to yellow house
        index_corners = (corners ~= n); %ID all numbers not equal to n
        corners = corners(index_corners); %keep just those numbers - 
                                          %problem may become if the number
                                          %was the 1st and last - but then
                                          %if lose too many, have < v opt
    end
    num_each_node = food_totals(counter) / (n-1); %amount of supplies for each node    
    if num_each_node > payload
        num_each_node = payload; %limit the amount taken to each node as one truckfull
                              %although of course if Yellow house went to
                              %a node multiple times, then the node could
                              %get multiple truckloads
                              
    end  
    ph_2 = floor(payload/num_each_node); %want to make sure we don't take too many loads
                                      %but the if statement above makes
                                      %sure we don't get ph_2 = 0
    v = ceil((n-1)/ph_2); %divide number of nodes by number of nodes which can be serviced by each vehicle
    %v = ceil(food_totals(counter)/1250); %number of vehicles needed - want to ROUND UP to the next vehicle

    %To find the corners of the hull to use for the seeds
    dv = floor((length(corners)-1)/v); 
    corners_sifted_1 = corners(1:dv:(end-1)); 
    if length(corners_sifted_1) > v
        corners_sifted = corners_sifted_1(1:v); 
    elseif length(corners_sifted_1) < v %we do have a problem with the convex hull - not always enough corners
        disp(['Problem with convex hull_' num2str(counter)])
        convex_hull_probs = [convex_hull_probs counter];
        ph_lat = linspace(27.307,29.781,v); %min and max from the R script
        ph_long = linspace(80.575, 87.328,v);
        corners_sifted(1:v,1) = ph_lat;
        corners_sifted(1:v,2) = ph_long;
        prob_with_hull = 1; %I've created enough seed points now
    elseif length(corners_sifted_1) == v %RECENTLY ADDED!  4/10/2017
        corners_sifted = corners_sifted_1;
    end
else
    v = 1; %Accounting for when only have one stop - so only need one vehicle
    prob_with_hull = 1; %we could also just say corners_sifted = 1 since just the first index
    corners_sifted(1,1) = lat(1);
    corners_sifted(2,1) = long(1);
end

%%%Getting Vehicle and Capacity Counts%%%
num_v(counter) = v;
num_each_node_all(counter) = num_each_node;

distance_matrix_1 = zeros(v,(n-1));

if prob_with_hull == 1 %need to create a new distance matrix with the new points
    for i = 1:v %for all of the 'corner' points - the row
        for j = 1:(n-1) %for all the nodes - the column
            distance_matrix_1(i,j) = norm(corners_sifted(i,:) - data(j,:));  
        end
    end
end

%Finding the distances between Yellow House and the Indices
distance_yh_seed = zeros(1,v);

for i = 1:v
    distance_yh_seed(i) = norm(corners_sifted(i,:)-[27.6825, 85.3059]);
end
    
%%%Actually creating the g matrix%%%

g = zeros(n-1,v); %need to calculate the additional cost of assigning the ith node
                  %to the kth vehicle

%%%%%%Finding the g matrix for the objective function%%%%%%%%%%
%corners_sifted has all of the indices that go with the vehicles k
if prob_with_hull == 0 %If there is not a problem with the hull, just use the distance matrix 
    for k = 1:v
        for i = 1:(n-1) %all the nodes (except for yellow house)
            g(i,k) = distance_matrix(n,i) + distance_matrix(i,corners_sifted(k)) - distance_matrix(n,corners_sifted(k)); 
        end
    end
else
    for k = 1:v   %finding the distances using the new seeds
        for i = 1:(n-1)
           g(i,k) = distance_matrix(n,i) + distance_matrix_1(k,i) - distance_yh_seed(k);                  
        end
    end
end

%%%%%%%%Setting up the Integer Linear Programming Constraints%%%%%%%%%%%

num_DV = v*(n-1); %number of decision variables

intcon = 1:num_DV; %getting number of variables for the integer program

f = g(1:(n-1),1)';

for i = 2:v
    f = [f g(1:(n-1),i)']; %arranged as ith job to kth, grouped by vehicle
                           %I do have enough numbers
end

f = abs(f); %ADDED RECENTLY 4/12/2017 12:01 AM


%Capacity constraints (= number of vehicles)
inequal_mats = cell(v,1);

for i = 1:v 
    placeholder = [zeros(1,(n-1)*(i-1)) num_each_node*ones(1,(n-1)) zeros(1,(n-1)*(v-i))];
    inequal_mats{i} = placeholder;
end

A = cell2mat(inequal_mats);
b = payload*ones(v,1); %right hand side for capacity constraint


%%%%%Assignment Constraints (= number of nodes)%%%%%%%

id = eye((n-1),(n-1));
Aeq = zeros((n-1),1);

for i = 1:v
   Aeq = [Aeq id]; %concatenate matrices together
end

Aeq = Aeq(1:(n-1), 2:end); %get RID OF FIRST COLUMN OF ZEROS

%%%%NEW: Adding in a constraint: 4/12/2017
%sum_constraint = ones(1,(n-1)*v);
%Aeq = [Aeq; sum_constraint];


% id = eye(16); 
% %other option would have been to concatenate 4 16x16 identity matrices
% Aeq = zeros(16,64);
% for i = 1:16 %number of nodes
%    Aeq(i,:) = [id(i,:) id(i,:) id(i,:) id(i,:)];  
% end

beq = ones((n-1),1);

%Adding extra beq for sum_constraint
%sum_constraints_b = (n-1);
%beq = [beq; sum_constraints_b];

%Binary Constraints
lb = zeros(num_DV,1);
ub = ones(num_DV,1);

%%%%Setting Up the Linear Program!!%%%%%%
%Structure according to https://www.mathworks.com/help/optim/ug/intlinprog.html#btxxm7t-4

%The steps intlinprog takes to solve: https://www.mathworks.com/help/optim/ug/mixed-integer-linear-programming-algorithms.html#bt6n8vs
%Look at the table here for options: https://www.mathworks.com/help/optim/ug/intlinprog.html?refresh=true#inputarg_options

%Integer Programming Options

disp(['****Output of Integer Program for Route ' num2str(counter) '*******']);
options = optimoptions('intlinprog', 'Heuristics', 'rins', 'IPPreprocess', 'none', 'LPPreprocess', 'none', 'NodeSelection', 'minobj');

sol = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,options); %need to investigate more
%always check the sum of sol is the number of nodes

%sol = int8(sol); %turning sol into integer

%Dealing with Matlab thinking 10^-10 should be a 1 when convert to logical
for i = 1:length(sol) %RECENTLY ADDED 12:21AM 4/13/2017
   if sol(i) < 10^-5 
        sol(i) = 0;
   end
end

sol_1 = sol;
sol = logical(sol); %trying thing where make sol logical

if sum(sol) ~= double(n-1) %seeing if only (n-1) nodes were assigned (don't want to double assign any nodes)
   disp('problem with assignment') 
   assignment_probs = [assignment_probs counter];
end


%%%%Graphing Which Nodes go with Which Vehicle%%%%%

sol = logical(sol); %getting the decision variables - turning sol into logical array
raw_sol{counter} = sol;

if isempty(sol) == 0
    node_assignments = cell(1,v);
    for i = 1:v
        node_assignments{i} = [data(sol((1+(n-1)*(i-1)):((n-1)*i)),:); [27.6825, 85.3059]]; %assigning coordinates to each vehicle         
    end
    all_assignments{counter} = node_assignments;
end

%Max number of vehicles using my dummy kg values is 18 vehicles - I can use
%the 8 colors available with just different symbols 

%sol = double(sol); %turning sol back into an integer

if (isempty(sol) == 0 && sum(sol) == double(n-1)) 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%END OF ASSIGNMENT STAGE%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%STAGE 2: Routing Among the Nodes%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Uncomment the section corresponding to the heuristic/metaheuristic you
%want to use for routing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%Greedy - Nearest Neighbor Algorithm%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%http://paginas.fe.up.pt/~mac/ensino/docs/OR/HowToSolveIt/ConstructiveHeuristicsForTheTSP.pdf
%%%http://faculty.washington.edu/jtenenbg/courses/342/f08/sessions/tsp.html
%%%http://www-e.uni-magdeburg.de/mertens/TSP/node2.html

% saving_number = 1;
% 
% index_vector = 1:(n-1);
% vehicle_indices = cell(1,v);
% 
% for i = 1:v
%    vehicle_indices{i} = [index_vector( sol((1+(n-1)*(i-1)):((n-1)*i)) ) n]; 
% end 
% 
% 
% node_ordering = cell(1,v);
% 
% for i = 1:v
%     vvec = vehicle_indices{1,i}; 
%     starting = vvec(end);
%     vvec = vvec(1:(end-1));
%     ordering = [starting]; %getting the ordering of the nodes
% while isempty(vvec) == 0 %while still entries in the vvec
%     mini_dist = zeros(1,length(vvec)); %distance between current node and canidates
%     for j = 1:length(vvec)
%        mini_dist(j) = distance_matrix(starting,vvec(j));
%     end
%     vvec_node_index = find(mini_dist == min(mini_dist),1); %find min of dists and the 
%                                                            %place in matrix since vvec and matrix
%                                                            %same length
%     ordering = [ordering vvec(vvec_node_index)]; %put index at end of ordering
%     starting = vvec(vvec_node_index); %need to establish new node 
%     vvec(vvec_node_index) = []; %get rid of index
% end
%     node_ordering{i} = [ordering n]; %for each vehicle, gets ordering
% end
% all_v_node_orderings{counter} = node_ordering;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%Simulated Annealing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%Set up Stuff - Getting the Storage Right

% saving_number = 2;
% index_vector = 1:(n-1);
% vehicle_indices = cell(1,v);
% 
% for i = 1:v
%    vehicle_indices{i} = [n index_vector( sol((1+(n-1)*(i-1)):((n-1)*i)) ) n]; %need the n in beginning for alg to work
% end %check_this
% 
% node_ordering = cell(1,v); %what I'm putting the orderings into
% 
% 
% for h = 1:v %for all vehicles, apply simulated annealing algorithm to route them
% 
% new_tour = vehicle_indices{h}; %original route is just assignments in order with ns on either side
%                                %n being the nth node which is yellow house
% T1 = 0.2 * tour_length_func(new_tour,distance_matrix); %use distances from original distance matrix
% T2 = 0.5*T1;
% T3 = 0.5*T2;
% T4 = 0.5*T3;
% T5 = 0.5*T4;
% T_candidates = [T1, T2, T3, T4, T5];
% tours_log = cell(1,num_each_temp*5); %tracking the tours
% length_tours_log = zeros(1,num_each_temp*5); %tracking the lengths - need to take the min of this and put in node_orderings 
% %checking_sum = zeros(1,15); %making sure all of the nodes are included in the route
% n_sa = length(new_tour); %n_sa is length of route including the repeat ns
%                          %for subtour reversal, might want to do -1 on the
%                          %length(new_tour)
% 
% 
%                          
% if n_sa > 4 %if there are actually more than 2 nodes that we need to travel to
%     
% for i = 1:5 %since 4 temperature schedules
% T = T_candidates(i);
% check_1 = ((i-1)*num_each_temp)+1; %put num_each_temp because this had a 3
% 
% while check_1 ~= (i*num_each_temp + 1)
%     check_1_1 = 1;
%     tour = new_tour; %want to make previous tour new tour
%     while check_1_1 ~= 0 %getting the indices for switching  
%         begin_slot = randi([2,(n_sa-2)],1); %not lst, last, or next-to-last
%         end_slot = randi([begin_slot+1,(n_sa-1)],1); %after beginning but not the last which is (n+1)
%         if (begin_slot == 2 && end_slot == (n_sa-1)) %if pick 2nd and nth, then when flip, wont do anything (just go backwrds)
%            check_1_1 = 1;
%         else
%            check_1_1 = 0;
%         end
%     end  
%     new_tour = tour( [1:(begin_slot-1) fliplr(begin_slot:end_slot) (end_slot+1):(n_sa)] ); %we are indexing into the tour
%     %need to remember that the length of [n 1:n] is actually n+1
%     z_old = tour_length_func(tour, distance_matrix);
%     z_new = tour_length_func(new_tour, distance_matrix);
%     
%     if z_new <= z_old
%         new_tour = new_tour; %new tour is new candidate!
%     else
%         prob_accept = exp((z_old - z_new)/T); 
%         move_selection_prob = rand; %generate a random number exclusively between 0 and 1
%         if prob_accept > move_selection_prob
%             new_tour = new_tour; %new tour is new candidate!
%         else
%             new_tour = tour; %tour is 'new' candidate - keep old candidate
%         end       
%     end
%     
%     length_tours_log(check_1) = tour_length_func(new_tour,distance_matrix);
%     tours_log{check_1} = new_tour; %keeping track of the current tour after each iteration
%     %checking_sum(check_1) = (sum(new_tour) == (sum(1:n)+n));
%     check_1 = check_1 + 1;
%     
% end
% 
% end
% 
% sa_min_index = find(length_tours_log == min(length_tours_log),1); %find the min of the lengths 
% node_ordering{h} = tours_log{sa_min_index};
% 
% else %if only one or two nodes that routing the vehicle to, then original path is right one
%     node_ordering{h} = new_tour;
% 
% 
% end
% 
% end
% 
% all_v_node_orderings{counter} = node_ordering; %putting the orderings into the larger data structure
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%Sub-Tour Reversal%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%Set up Stuff - Getting the Storage Right

saving_number = 3;
index_vector = 1:(n-1);
vehicle_indices = cell(1,v);

for i = 1:v
   vehicle_indices{i} = [n index_vector( sol((1+(n-1)*(i-1)):((n-1)*i)) ) n]; %need the n in beginning for alg to work
end %check_this

node_ordering = cell(1,v); %what I'm putting the orderings into

%%%%Algorithm%%%%%
for h = 1:v %for all vehicles

route = vehicle_indices{h}; %original route is just assignments in order with ns on either side
                               %n being the nth node which is yellow house
no_min = 0;
n_sr = (length(route) - 1); %want the actual number of nodes + yellow house 
                            %dont want to double count yellow house
                           
                            
if n_sr > 3 %more than 2 nodes + yellow house
while no_min ~= 1

counter_sr = (n_sr-2); %this is the largest window we can use
routes_stored = {0};
lengths_stored = 1;
indexing_for_routes_stored = 1;

current_route_length = tour_length_func(route, distance_matrix);

while counter_sr ~= 1 
   num_times_use_windows = n_sr-counter_sr; %counter_sr initially is longest window, so we are getting the number of times
                            %that this specific window can be used
   routes = cell(1,num_times_use_windows); %routes for the specific window
   length_vec = zeros(1,num_times_use_windows);
   
   for i = 2:(num_times_use_windows+1) %we want to go num_times_use_windows, but want to shift it 

       indices = [1:(i-1) fliplr(i:(i+(counter_sr-1))) (i+counter_sr):(n_sr+1)]; 
                                                    
       route_ph = route(indices);
       routes{i-1} = route_ph; %the route for each window
       length_vec(i-1) = tour_length_func(route_ph,distance_matrix); %length of the vector
        
   end
   routes_stored = [routes_stored, routes]; 
   lengths_stored = [lengths_stored length_vec]; 
   
   indexing_for_routes_stored = indexing_for_routes_stored + 1;
   counter_sr = counter_sr - 1;
end %finished going through all subtour reversals of k=2 for this trial route

lengths_stored = lengths_stored(2:end);
routes_stored = routes_stored(2:end);

min_index = find(lengths_stored == min(lengths_stored),1);
route_1 = routes_stored{min_index}; %lengths_stored and routes_stored are calibrated with each other

if current_route_length <= tour_length_func(route_1,distance_matrix)
    no_min = 1; %if not less than current length, then STOP
else
    route = route_1; %if less than current, do this again
end

end

node_ordering{h} = route;

else
    node_ordering{h} = route;
    
end


end

%%Putting the orderings into the larger data structure
all_v_node_orderings{counter} = node_ordering; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%End of Stage 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

check_assignment = n;
for i = 1:v
    ph_10 = node_ordering{i};
    check_assignment = [check_assignment ph_10(2:(end-1))];   
end

check_assignment = sort(check_assignment); %didn't need to sort
ck_routing_sorted{counter} = check_assignment;
check_assignment_1 = sum(check_assignment);
ck_1 = sum(1:n);

if check_assignment_1 == ck_1
    checking_routing{counter} = 'yes!';
else
    checking_routing{counter} = 'no'; %not all nodes assigned    
end

%%%Counting the Total Distance 

each_route_dist = zeros(1,v);

for i = 1:length(node_ordering)
    ph_33 = node_ordering{i};
    %tic
    each_route_dist(i) = tour_length_func(ph_33, distance_matrix);
    %toc
end

total_distance_each_route{counter} = each_route_dist; %distance for each route in each day
total_distance_each_day(counter) = sum(each_route_dist); %total distance traveled each day by all vehicles


end

counter = counter + 1;
end
toc

%%%Saving Stuff to Files%%%%
if saving_number == 1
   save(sprintf('greedy_alg_distances_%d.mat',payload), 'total_distance_each_day'); %might have to add -mat
   save(sprintf('all_v_node_orderings_greedy_%d.mat',payload),'all_v_node_orderings'); 
elseif saving_number == 2
   save(sprintf('simulated_annealing_distances_%d.mat',payload), 'total_distance_each_day'); 
   save(sprintf('all_v_node_orderings_sim_annealing_%d.mat',payload),'all_v_node_orderings');
elseif saving_number == 3
   save(sprintf('subtour_reversal_distances_%d.mat',payload), 'total_distance_each_day'); 
   save(sprintf('all_v_node_orderings_subtour_%d.mat',payload),'all_v_node_orderings');
end

%%%%Saving all assignments to files to make side by side graphs%%%%
save(sprintf('all_assignments_%d.mat',payload), 'all_assignments');



%%
%%%%%%%%Creating all the Figures%%%%%%%%%
%all_assignments{counter} = node_assignments; (node assignments are the 

%%%Number of Nodes versus Total Distance Traveled%%%
figure
subplot(1,2,1)
plot(number_nodes,total_distance_each_day,'*')
title('Number of Nodes versus Total Distance Traveled')
xlabel('Number of Nodes')
ylabel('Total Distance Traveled')
%%%Number of Vehicles versus Total Distance Traveled%%%
subplot(1,2,2)
plot(num_v,total_distance_each_day,'m*')
title('Number of Vehicles versus Total Distance Traveled')
xlabel('Number of Vehicles')
ylabel('Total Distance Traveled')

%%%%%%Assignments for Each Day%%%%%%%%%%%
vec_colors = {'m+', 'g+', 'b+', 'k+', 'mo', 'go', 'bo', 'ko', 'm^', 'g^', 'b^', 'k^', 'm*', 'g*', 'b*', 'k*', 'mx', 'gx', 'bx', 'kx', 'md', 'gd', 'bd', 'kd'};

%Days 1-8%
figure
for j = 1:8 %just copy and paste this a few times
    subplot(2,4,j) %creating a new figure
    placeholder_3 = all_assignments{j}; %placeholder_3 - cell array with each vehicle
for i = 1:num_v(j)
    placeholder_2 = placeholder_3{i};
    hold on
    plot(placeholder_2(:,1),placeholder_2(:,2),vec_colors{i})
    hold on
    plot(27.6825, 85.3059,'ys') %this won't be a problem since it will keep plotting over itself (?)
end
    axis([27.307,28.5,84.25,87.328]) %[27.307,29.781,80.575,87.328] (mins and maxes)
    title([sprintf('Allocation of Nodes to \n the Vehicles on Day ') num2str(j)])
    xlabel('Latitude')
    ylabel('Longitude')
end

%Days 9-16%
figure
for j = 1:8 %day - retrieving day information
    subplot(2,4,j) %creating a new figure
    placeholder_3 = all_assignments{j+8}; %placeholder_3 - cell array with each vehicle
for i = 1:num_v(j+8) %retrieving route i information from the jth day 
    placeholder_2 = placeholder_3{i};
    hold on
    plot(placeholder_2(:,1),placeholder_2(:,2),vec_colors{i})
    hold on
    plot(27.6825, 85.3059,'ys') %this won't be a problem since it will keep plotting over itself (?)
end
    axis([27.307,28.5,84.25,87.328]) %[27.307,29.781,80.575,87.328] (mins and maxes)
    title([sprintf('Allocation of Nodes to \n the Vehicles on Day ') num2str(j+8)])
    xlabel('Latitude')
    ylabel('Longitude')
end

%Days 17-24%
figure
for j = 1:8 %day - retrieving day information
    subplot(2,4,j) %creating a new figure
    placeholder_3 = all_assignments{j+16}; %placeholder_3 - cell array with each vehicle
for i = 1:num_v(j+16) %retrieving route i information from the jth day 
    placeholder_2 = placeholder_3{i};
    hold on
    plot(placeholder_2(:,1),placeholder_2(:,2),vec_colors{i})
    hold on
    plot(27.6825, 85.3059,'ys') %this won't be a problem since it will keep plotting over itself 
end
    axis([27.307,28.5,84.25,87.328]) %[27.307,29.781,80.575,87.328] (mins and maxes)
    title([sprintf('Allocation of Nodes to \n the Vehicles on Day ') num2str(j+16)])
    xlabel('Latitude')
    ylabel('Longitude')
end

%Days 25-26%
figure
for j = 1:2 %day - retrieving day information
    subplot(2,4,j) %creating a new figure
    placeholder_3 = all_assignments{j+24}; %placeholder_3 - cell array with each vehicle
for i = 1:num_v(j+24) %retrieving route i information from the jth day 
    placeholder_2 = placeholder_3{i};
    hold on
    plot(placeholder_2(:,1),placeholder_2(:,2),vec_colors{i})
    hold on
    plot(27.6825, 85.3059,'ys') %this won't be a problem since it will keep plotting over itself (?)
end
    axis([27.307,28.5,84.25,87.328]) %[27.307,29.781,80.575,87.328] (mins and maxes)
    title([sprintf('Allocation of Nodes to \n the Vehicles on Day ') num2str(j+24)])
    xlabel('Latitude')
    ylabel('Longitude')
end


%%%%%%%%Making Graph for Presentation%%%%%%%%%
%Days 17-24%
figure
for j = 1:4 %day - retrieving day information
    subplot(2,2,j) %creating a new figure
    placeholder_3 = all_assignments{j+18}; %placeholder_3 - cell array with each vehicle
for i = 1:num_v(j+18) %retrieving route i information from the jth day 
    placeholder_2 = placeholder_3{i};
    hold on
    plot(placeholder_2(:,1),placeholder_2(:,2),vec_colors{i})
    hold on
    plot(27.6825, 85.3059,'ys') %this won't be a problem since it will keep plotting over itself (?)
end
    axis([27.307,28.2,84.25,86.5]) %[27.307,29.781,80.575,87.328] (mins and maxes)
    title([sprintf('Allocation of Nodes to \n the Vehicles on Day ') num2str(j+18)])
    xlabel('Latitude')
    ylabel('Longitude')
end





