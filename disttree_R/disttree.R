########################################################################
# new version of disttree using extree directly without applying ctree #
########################################################################

disttree <- function(formula,
                       data,
                       subset,
                       na.action = na.pass,
                       weights,
                       offset,
                       cluster,
                       family = NO(),
                       control = disttree_control(...),
                       converged = NULL,
                       scores = NULL,
                       doFit = TRUE,
                       ...) {

  ## Clean up control
  type.hessian <- control$type.hessian
  decorrelate <- control$decorrelate
  method <- control$method
  optim.control <- control$optim.control
  lower <- control$lower
  upper <- control$upper

  ocontrol <- control
  control$type.hessian <- control$decorrelate <- control$method <- control$optim.control <- NULL
  control$lower <- control$upper <- NULL
  if(control$saveinfo == FALSE){
    control$saveinfo <- TRUE
    warning("'control$saveinfo' set to TRUE, needed for distributional tree.")
  }

  ## Keep call
  cl <- match.call(expand.dots = TRUE)
  if(missing(data)) data <- environment(formula)

  # Check formula
  oformula <- as.formula(formula)
  formula <- Formula::as.Formula(formula)
  if(length(formula)[2L] > 1L) {
    formula <- Formula::Formula(formula(formula, rhs = 2L))
    # NOTE: (LS) if rhs has more than 1 element it is here assumed that partitioning variables are handed over on 2nd slot
    warning("formula must not have more than one RHS parts (only partitioning variables allowed)")
  }

  ## Set up model.frame() call
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "na.action", "weights",
               "offset", "cluster", "scores"), names(mf), 0L)

  mf <- mf[c(1L, m)]
  mf$yx <- "matrix"
  mf$nmax <- control$nmax
  mf$ytype <- control$ytype

  ## Evaluate model.frame
  mf[[1L]] <- quote(partykit::extree_data)

  d <- eval(mf, parent.frame())

  subset <- partykit:::.start_subset(d)
  weights <- model.weights(model.frame(d))

  if (is.null(control$update)) control$update <- TRUE

  # Set up family
  if(!inherits(family, "disttree.family"))
    family <- distfamily(family)

  #Y <- d$yx[[1]]
  # (LS) check whether d$yx really contains only the response (and no covariates)
  if(length(d$yx) > 1) stop("covariates can only be used as split variables")  # NOTE: (ML) can this happen after formula creation?

  if(NCOL(d$yx[[1]]) > 1) stop("response variable has to be univariate") # TODO: (ML) adapt for multidimensional responses
  if(inherits(d$yx[[1]], "interval")) stop("can not deal with binned intervals yet")

  ## Set up wrapper function for distfit
  ytrafo <- function(subset, weights, estfun = FALSE, object = FALSE, info = NULL) {

    ys <- d$yx[[1]][subset]  # necessary to get response data into the function
    subweights <- if(is.null(weights) || (length(weights)==0L)) weights else weights[subset]

    # model <- disttree::distfit(ys, family = family, weights = subweights, start = info$coefficients, start.eta = NULL,
    #                              vcov = (decorrelate == "vcov"), type.hessian = type.hessian,
    #                              method = method, estfun = estfun, optim.control = optim.control,
    #                              lower = lower, upper = upper)

    model <- distfit(ys, family = family, weights = subweights, start = info$coefficients, start.eta = NULL,
                                 vcov = (decorrelate == "vcov"), type.hessian = type.hessian,
                                 method = method, estfun = estfun, optim.control = optim.control,
                                 lower = lower, upper = upper)


    if(estfun) {
      ef <- as.matrix(model$estfun) # distfit returns weighted scores!

      if(decorrelate != "none") {
        n <- NROW(ef)
        ef <- ef/sqrt(n)

        vcov <- if(decorrelate == "vcov") {
          vcov(model, type = "link") * n
        } else {
          solve(crossprod(ef))
        }

        root.matrix <- function(X) {
          if((ncol(X) == 1L)&&(nrow(X) == 1L)) return(sqrt(X)) else {
            X.eigen <- eigen(X, symmetric = TRUE)
            if(any(X.eigen$values < 0)) stop("Matrix is not positive semidefinite")
            sqomega <- sqrt(diag(X.eigen$values))
            V <- X.eigen$vectors
            return(V %*% sqomega %*% t(V))
          }
        }
        ef <- as.matrix(t(root.matrix(vcov) %*% t(ef)))
      }

      estfun <- matrix(0, ncol = ncol(ef), nrow = nrow(d$data))
      estfun[subset,] <- ef
      ### now automatically if(!(is.null(weights) || (length(weights)==0L))) estfun <- estfun / weights
    } else {
      estfun <- NULL
    }
    rval <- list(estfun = estfun,
                unweighted = FALSE, # (LS) unweighted = TRUE would prevent estfun / w in extree_fit
                coefficients = coef(model, type = "parameter"),
                objfun = -logLik(model),  # (LS) optional function to be minimized
                object = if(object) model else NULL,
                nobs = nobs(model), # TODO: (ML) check if ok, added to get nobs right
                converged = model$converged  # TODO: (LS) warnings if distfit does not converge
    )

    return(rval)
  }

  ## Set up default 'converged' function
  # checking whether all response values are equal in one node, if so return FALSE
  converged_default <- function(data, weights, control){
    #if(is.null(weights)) weights <- rep.int(1, NROW(data$yx[[1]]))
    convfun <- function(subset, weights){
      ys <- data$yx[[1]][subset]
      ws <- if(is.null(weights) || (length(weights)==0L)) rep.int(1, NROW(ys)) else weights[subset]
      conv <- length(unique(ys[ws > 0])) > 1
      return(conv)
    }
    return(convfun)
  }

  if (is.function(converged)) {
    stopifnot(all(c("data", "weights", "control") %in% names(formals(converged))))
    converged <- converged(d, weights, control = control) &
      converged_default(d, weights, control = control)
  } else {
    converged <- converged_default(d, weights, control = control)
  }

  ## Set up wrapper for extree_fit with predefined fit function
  update <- function(subset, weights, control, doFit = TRUE) {
    partykit::extree_fit(data = d, trafo = ytrafo, converged = converged, partyvars = d$variables$z,
                         subset = subset, weights = weights, ctrl = control, doFit = doFit)
  }

  if (!doFit) return(list(d = d, update = update))

  ## Set minsize to 10 * number of parameters, if NULL ## TODO: (ML) n_coef could be get from family?!
  if (is.null(control$minbucket) | is.null(control$minsplit)) {
      ctrl <- control
      N <- sum(complete.cases(model.frame(d, yxonly = TRUE)))
      ctrl$minbucket <- ctrl$minsplit <- N
      ctrl$logmincriterion <- Inf
      ctrl$stump <- TRUE

      tree <- update(subset = subset, weights = weights, control = ctrl)
      cf <- tree$trafo(subset = subset, weights = weights, info = NULL)$coefficient
      if (is.null(cf)) {
          n_coef <- 1
      } else {
          n_coef <- length(cf)
      }
      minsize <- as.integer(ceiling(10L * n_coef / NCOL(d$yx$y)))
      if (is.null(control$minbucket)) control$minbucket <- minsize
      if (is.null(control$minsplit)) control$minsplit <- minsize
  }

  ## Call the actual workhorse
  tree <- update(subset = subset, weights = weights, control = control)
  trafo <- tree$trafo

  ### Prepare as modelparty/constpary
  mf <- model.frame(d)
  if(is.null(weights) || (length(weights)==0L)) weights <- rep(1, nrow(mf))

  fitted <- data.frame("(fitted)" = partykit::fitted_node(tree$nodes, mf),
                       "(weights)" = weights,
                       check.names = FALSE)

  fitted[[3]] <- y <- mf[, d$variables$y, drop = TRUE] # NOTE: (ML) y added, necessary?
  names(fitted)[3] <- "(response)"

  control$ytype <- ifelse(is.vector(y), "vector", class(y)) # NOTE: (ML) needed? (from mob)
  control$xtype <- "matrix" # TODO: (AZ) find out when to use data.frame NOTE: (ML) needed (from mob)

  rval <- partykit::party(tree$nodes,
                data = if(control$model) mf else mf[0,],
                fitted = fitted,
                terms = d$terms$all,
                info = list(
                  call = cl,
                  formula = formula,
                  family = family,
                  fit = distfit,
                  #nobs = nrow(fitted),
                  control = ocontrol
              )
  )

  class(rval) <- c("modelparty", class(rval))  # FIXME: either model or constparty object!
  # TODO: (AZ) check if this can be done prettier
  which_terminals <- partykit::nodeids(rval, terminal = TRUE)
  which_all <- partykit::nodeids(rval)

  idx <- lapply(which_all, partykit:::.get_path, obj = tree$nodes)
  names(idx) <- which_all
  tree_ret <- unclass(rval)
  subset_term <- predict(rval, type = "node")

  for (i in which_all) {

    ichar <- as.character(i)
    iinfo <- tree_ret[[c(1, idx[[ichar]])]]$info

    if (i %in% which_terminals) winfo <- control$terminal else
      winfo <- control$inner

    if (is.null(winfo)) {
      iinfo$object <- NULL
      iinfo$estfun <- NULL
    } else {
      if (is.null(iinfo) | any(is.null(iinfo[[winfo]])) |
          any(! winfo %in% names(iinfo))) {
        iinfo <- trafo(subset = which(subset_term == i), weights = weights, info = NULL,
                       estfun = ("estfun" %in% winfo),
                       object = ("object" %in% winfo))
      }
    }

    tree_ret[[c(1, idx[[ichar]])]]$info <- iinfo
  }

  tree_ret$update <- update

  class(tree_ret) <- c("disttree", class(rval))

  return(tree_ret)

}


disttree_control <- function(type.tree = NULL, #c("mob", "ctree", "guide"),
                               type.hessian = c("checklist", "analytic", "numeric"),
                               decorrelate = c("none", "opg", "vcov"),
                               method = "L-BFGS-B",
                               optim.control = list(),
                               lower = -Inf,
                               upper = Inf,
                               minsplit = NULL,     # NOTE: (ML) currently use mob default
                               minbucket = NULL,    # NOTE: (ML) currently use mob default
                               splittry = 1L,       # NOTE: (ML) currently use mob default

                               ## Arguments need for disttree
                               splitflavour = c("ctree", "exhaustive"),
                               testflavour = c("ctree", "mfluc", "guide"),
                               terminal = "object",
                               model = TRUE,
                               inner = "object",

                               ## Additional arguments for exhaustive/mobster
                               restart = TRUE,
                               breakties = FALSE,
                               parm = NULL,
                               dfsplit = TRUE,
                               vcov = c("opg", "info", "sandwich"),
                               ordinal = c("chisq", "max", "L2"),
                               ytype = c("vector", "data.frame", "matrix"),
                               trim = 0.1,
                               #nrep = 10000L,       #FIXME: (ML) Is in mob included, needed for dt?
                               #catsplit = "binary", #FIXME: (ML) Is in mob included, needed for dt?
                               #numsplit = "left",   #FIXME: (ML) Is in mob included, needed for dt?
                               #minsize = NULL,      #FIXME: (ML) Is in mob included, needed for dt?
                               #minprob = 0.01,      #FIXME: (ML) Is in mob included, needed for dt?
                               #nmax = Inf,          #FIXME: (ML) Is in mob included, needed for dt?

                               ## Additonal arguments for GUIDE
                               guide_interaction = FALSE,
                               interaction = FALSE,
                               #guide_unweighted = FALSE,
                               guide_parm = NULL,  # a vector of indices of the parameters (incl. intercept) for which estfun should be considered
                               guide_testtype = c("max", "sum", "coin"),
                               guide_decorrelate = "vcov",   # needs to be set to other than "none" for testtype max and sum
                               xgroups = NULL,  # number of categories for split variables (optionally breaks can be handed over)
                               ygroups = NULL,  # number of categories for scores (optionally breaks can be handed over)
                               weighted.scores = FALSE,   # logical, should scores be weighted in GUIDE
                               ...) {

  ctrl <- partykit::ctree_control(minsplit = minsplit, minbucket = minbucket, splittry = splittry, ...)

  ## Add parameters needed to return coefficient, etc.
  ctrl$terminal <- terminal
  ctrl$model <- model
  ctrl$inner <- inner

  ## Add additional parameters needed within disttree
  ctrl$type.hessian <- match.arg(type.hessian)
  ctrl$decorrelate <- match.arg(decorrelate)
  ctrl$method <- method
  ctrl$optim.control <- optim.control
  ctrl$lower <- lower
  ctrl$upper <- upper
  ctrl$dfsplit <- dfsplit  # FIXME: (ML) Added to all types of tree, to get df within logLik.modelparty

  ## Check the kind of tree
  if (length(testflavour) == 3 & is.null(type.tree)) testflavour <- testflavour[1]
  if (length(splitflavour) == 2 & is.null(type.tree)) splitflavour <- splitflavour[1]

  if(!is.null(type.tree)) {

    if(!type.tree %in% c("ctree", "mob")) {
      stop("type.tree can only be set to 'ctree' or 'mob'")
    } else {

      if(type.tree == "ctree"){
        if(!("ctree" %in% splitflavour & "ctree" %in% testflavour)){
          stop("for type.tree = 'ctree' testflavour and splitflavour can not be set to other than 'ctree'")
        } else {

          testflavour <- "ctree"
          splitflavour <- "ctree"

        }
      }

      if(type.tree == "mob"){
        if(!("exhaustive" %in% splitflavour & "mfluc" %in% testflavour)){
          stop("for type.tree = 'mob' testflavour can not be set to other than 'mfluc' and splitflavour can not be set to other than 'exhaustive'")
        } else {

          testflavour <- "mfluc"
          splitflavour <- "exhaustive"

        }
      }
    }
  }

  if(testflavour == "mfluc" | splitflavour == "exhaustive") {
    ctrl <- c(ctrl, list(restart = restart,
                         breakties = breakties,
                         parm = parm,
                         #dfsplit = dfsplit,  FIXME: (ML) Added to all tree types, see comment above
                         vcov = vcov,
                         ordinal = match.arg(ordinal),
                         ytype = ytype,
                         trim = trim))
  }

  if(testflavour == "guide") {

    if(!(ctrl$criterion == "p.value") && guide_interaction){
      stop("For testflavour GUIDE with interaction tests only 'p.value' can be selected as criterion")
    }
    ctrl <- c(ctrl, list(guide_parm = guide_parm,    # LS: a vector of indices of parameters for which estfun should be considered
              guide_testtype = guide_testtype,
              interaction = interaction,
              guide_decorrelate = guide_decorrelate, # (LS) needs to be set to other than "none" for testtype max and sum
                                                     # (LS) unless ytrafo returns decorrelated scores
                                                     # FIXME: (LS) c("none","vcov","opg")
              xgroups = xgroups,                     # (LS) number of categories for split variables (optionally breaks can be handed over)
              ygroups = ygroups,                     # (LS) number of categories for scores (optionally breaks can be handed over)
              weighted.scores = weighted.scores))    # (LS) logical, should scores be weighted
  }

  ## Overwrite the split- and selectfun according to the split- and testflavour
  if (splitflavour == "ctree") ctrl$splitfun <- partykit:::.ctree_split()
  if (splitflavour == "exhaustive") ctrl$splitfun <- partykit:::.objfun_split()

  if (testflavour == "ctree") ctrl$selectfun <- partykit:::.ctree_select()
  if (testflavour == "mfluc") ctrl$selectfun <- partykit:::.mfluc_select()
  if (testflavour == "guide") ctrl$selectfun <- .guide_select()

  # FIXME: (LS) argumets numsplit in mob and intersplit in ctree/extree
  # intersplit <- numsplit == "center"

  return(ctrl)
}



## FIXME: adapt methods (class disttree?)
print.disttree <- function(x, title = NULL, objfun = "negative log-likelihood", ...)
{
  familyname <- if(inherits(x$info$family, "gamlss.family")) {
    paste(x$info$family[[1]][2], "Distribution")
  } else {x$info$family$family.name}
  if(is.null(title)) title <- sprintf("Distributional regression tree (%s)", familyname)
  partykit::print.modelparty(x, title = title, objfun = objfun, ...)
}



predict.disttree <- function (object, newdata = NULL, type = c("parameter", "response", "node"), OOB = FALSE, ...)
{

  # per default 'type' is set to 'parameter'
  type <- match.arg(type)

  ## get nodes
  ## if ctree was applied   # FIXME: currently class constparty can not be obtained from disttree
  #if(inherits(object, "constparty")) pred.nodes <- partykit::predict.party(object, newdata =  newdata,
  #                                                                         type = "node", OOB = OOB, ...)
  # if mob was applied
  if(inherits(object, "modelparty")) pred.nodes <- partykit::predict.modelparty(object, newdata =  newdata,
                                                                                type = "node", OOB = OOB, ...)

  if(type == "node") return(pred.nodes)

  ## get parameters
  groupcoef <- coef(object)

  # only 1 subgroup or 1-parametric family
  if(is.vector(groupcoef)) {
    # 1-parametric family
    if(length(object$family$link) == 1){
      groupcoef <- as.matrix(groupcoef)
      colnames(groupcoef) <- "mu"
    } else {
      # only 1 subgroup
      groupcoef <- t(as.matrix(groupcoef))
      rownames(groupcoef) <- "1"
    }
  }

  pred.par <- groupcoef[paste(pred.nodes),]

  if(is.vector(pred.par)){
    # 1-parametric family
    if(length(object$family$link) == 1) {
      pred.par <- as.matrix(pred.par)
      colnames(pred.par) <- "mu"
    } else {
      # only 1 new observation
      pred.par <- t(as.matrix(pred.par))
    }
  }

  rownames(pred.par) <- c(1: (NROW(pred.par)))
  pred.par <- as.data.frame(pred.par)

  if(type == "parameter") return(pred.par)

  if(type == "response") return(get_expectedvalue(object, pred.par))
}



coef.disttree <- function(object, ...){
  partykit:::coef.modelparty(object)
}


fitted.disttree <- function(object, ...){

  rval <- predict.disttree(object, newdata = NULL, type = "response", OOB = FALSE, ...)
  return(rval)

}

## FIXME: Should we allow for newdata in logLik ?
logLik.disttree <- function(object, newdata = NULL, weights = NULL, ...) {
  if(is.null(newdata)) {
    if(!is.null(weights)) stop("for weighted loglikelihood hand over data as newdata")
    if(!is.null(object$loglik)) return(structure(object$loglik, df = ncol(coef(object))*width(object) + width(object)-1 , class = "logLik"))
    newdata <- object$data
  }

  if(!is.null(weights)) stopifnot(NROW(newdata) == nrow(weights))
  ll <- 0
  # predicted nodes for the new dataset
  pred.node <- predict(object, newdata = newdata, type = "node")
  # coefficients in the terminal nodes
  coef_tn <- coef(object)
  # number of terminal nodes
  n_tn <- width(object) # <- nrow(coef_tn)
  # id of terminal nodes
  id_tn <- rownames(coef_tn)
  # get link fun and ddist from distribution list
  linkfun <- object$info$family$linkfun
  ddist <- object$info$family$ddist


  if(object$info$family$gamlssobj && object$info$family$censored) {
    censtype <- object$info$censtype
    censpoint <- object$info$censpoint
    for(i in 1:n_tn){
      par <- coef_tn[i,]
      eta <-  as.numeric(linkfun(par))
      # response variable and weights of the observations that end up in this terminal node
      nobs_tn <- newdata[pred.node == id_tn[i], paste(object$info$formula[[2]])]
      weights_tn <- if(!is.null(weights)) weights[pred.node == id_tn[i]] else rep.int(1, length(nobs_tn))
      if(length(nobs_tn) > 0L){
        if(!survival::is.Surv(nobs_tn)) {
          if(censtype == "left") ll <- ll + ddist(survival::Surv(nobs_tn, nobs_tn > censpoint, type = "left"), eta = eta, log = TRUE, sum = TRUE, weights = weights_tn)
          if(censtype == "right") ll <- ll + ddist(survival::Surv(nobs_tn, nobs_tn < censpoint, type = "right"), eta = eta, log = TRUE, sum = TRUE, weights = weights_tn)
          ## FIX ME: interval censored
        } else ll <- ll + ddist(nobs_tn, eta = eta, log=TRUE, sum = TRUE)
      }
    }
  } else {
    for(i in 1:n_tn){
      par <- coef_tn[i,]
      eta <-  as.numeric(linkfun(par))
      # response variable and weights of the observations that end up in this terminal node
      nobs_tn <- newdata[pred.node == id_tn[i], paste(object$info$formula[[2]])]
      weights_tn <- if(!is.null(weights)) weights[pred.node == id_tn[i]] else rep.int(1, length(nobs_tn))
      if(length(nobs_tn) > 0L) ll <- ll + ddist(nobs_tn, eta = eta,  log=TRUE, sum = TRUE, weights = weights_tn)
    }
  }
  return(structure(ll, df = ncol(coef(object))*width(object) + width(object)-1 , class = "logLik"))
}



if(TRUE) {  ##TODO: (ML) Needed for guide, delete if guide tests should not be included
# use different version of .select than partykit:::select for GUIDE as selectfun
# (returns p.values from curvature and interaction tests instead of p.value and teststatistic)
.select_g <- function(model, trafo, data, subset, weights, whichvar, ctrl, FUN) {
  ret <- list(criteria = matrix(NA, nrow = 2L, ncol = ncol(model.frame(data))))
  rownames(ret$criteria) <- c("statistic", "p.value")
  colnames(ret$criteria) <- names(model.frame(data))
  if (length(whichvar) == 0) return(ret)
  ### <FIXME> allow joint MC in the absense of missings; fix seeds
  ### write ctree_test / ... with whichvar and loop over variables there
  ### </FIXME>
  for (j in whichvar) {
    tst <- FUN(model = model, trafo = trafo, data = data,
               subset = subset, weights = weights, j = j,
               SPLITONLY = FALSE, ctrl = ctrl)
    ret$criteria["statistic",j] <- - as.numeric(tst["p.curv"])
    ret$criteria["p.value",j] <- - as.numeric(tst["p.min"])

    # for testflavour = "guide" only "p.value" can be chosen as criterion
    # because 2 p.values need to be returned (from curvature test and the min from curvature and interaction test)
    # the min is stored as 'p.value' in the returned matrix
    # if this min is from an interaction test, two covariates will have the same p.value
    # in this case the one with the lower p.value from the curvature test should be chosen
    # in extree: among the two covariates with the lowest p.value (highest negativ p.value) the one with the higher test statistic is chosen (ranked)
    # -> the negative p.value from the curvature test is stored as "statistic"
  }
  ret
}


.guide_select <- function(...)
  function(model, trafo, data, subset, weights, whichvar, ctrl) {
    args <- list(...)
    ctrl[names(args)] <- args
    .select_g(model, trafo, data, subset, weights, whichvar, ctrl, FUN = .guide_test)   # optional: use partykit:::.select
  }

.guide_test <- function(model, trafo, data, subset, weights, j, SPLITONLY = FALSE, ctrl) {

  ## TO DO: include SPLITONLY, MIA, ... ?
  ix <- data$zindex[[j]] ### data[[j, type = "index"]]
  iy <- data$yxindex ### data[["yx", type = "index"]]
  Y <- model$estfun  ## model from distfit, returns wheigthed scores
  if(ctrl$guide_unweighted) Y <- Y/weights  ## FIX ME: influence of weights only on categorization
  x <- data[[j]]
  if(!is.null(subset)) {
    Y <- if(is.vector(Y)) Y[subset] else Y[subset,]
    x <- x[subset]
  }

  # if all values of the selected covariate are equal return highest possible p.value
  if(length(unique(x))<2) return(c(p.min = 1, p.curv = 1))

  # only select those other covariates which are also in partyvars
  ix_others <- c(1:NCOL(data$data))[data$variables$z + ctrl$partyvars == 2]
  ix_others <- ix_others[!ix_others == j]

  # split Y into 2 parts based on whether residuals (here: scores) are positive or negative
  # separately for each parameter
  for(k in 1:model$object$npar){
    respos <- (Y[,k]>0)
    #respos <- factor((Y[,k]>0), levels = c(FALSE,TRUE), labels = c(0,1))
    Y <- cbind(Y, respos)
    colnames(Y)[(model$object$npar + k)] <- paste0("rp",k)
  }

  if(is.numeric(x)){
    x_cat <- rep.int(1, length(x))
    q1 <- quantile(x, 0.25)
    q2 <- quantile(x, 0.50)
    q3 <- quantile(x, 0.75)
    for(l in 1: length(x)){
      if(x[l] > q1 & x[l] <= q2) x_cat[l] <- 2
      if(x[l] > q2 & x[l] <= q3) x_cat[l] <- 3
      if(x[l] > q3) x_cat[l] <- 4
    }
    x_cat <- factor(x_cat, levels = c(1:4))
  } else {
    x_cat <- x
  }

  ## compute curvature test (for each parameter separately)
  p.curv <- chisq.test(x = x_cat, y = Y[,(model$object$npar+1)])$p.value
  if(model$object$npar > 1){
    for(k in 2:model$object$npar){
      p <- chisq.test(x = x_cat, y = Y[,(model$object$npar+k)])$p.value
      if(p < p.curv) p.curv <- p
    }
  }

  p.min <- p.curv

  ## compute interaction test (for each parameter and for each of the other covariates separately)
  # only keep test if p.value is smaller than the one resulting from the curvature test
  if(ctrl$guide_interaction & length(ix_others)>0){

    for(v in ix_others){

      xo <- data[[v]]
      if(!is.null(subset)) xo <- xo[subset]

      # only consider other covariate for interaction test if not all of its values are equal
      if(length(unique(xo))>1){

        x_cat_2d <- rep.int(1, length(x))

        if(is.factor(x) & is.factor(xo)){
          c1 <- length(levels(x))
          c2 <- length(levels(xo))
          for(l in 1:length(x)){
            for(m in 1:c1){
              for(n in 1:c2){
                if(x[l] == levels(x)[m] & xo[l] == levels(xo)[n]) x_cat_2d[l] <- (m-1)*c2+n
              }
            }
          }
        }

        if(is.factor(x) & is.numeric(xo)){
          c1 <- length(levels(x))
          med_xo <- median(xo)
          for(l in 1:length(x)){
            for(m in 1:c1){
              if(x[l] == levels(x)[m]){
                x_cat_2d[l] <- if(xo[l] > med_xo) c1+m else m
              }
            }
          }
        }

        if(is.numeric(x) & is.factor(xo)){
          c2 <- length(levels(xo))
          med_x <- median(x)
          for(l in 1:length(x)){
            for(n in 1:c2){
              if(xo[l] == levels(xo)[n]){
                x_cat_2d[l] <- if(x[l] > med_x) c2+n else n
              }
            }
          }
        }

        if(is.numeric(x) & is.numeric(xo)){
          med_x <- median(x)
          med_xo <- median(xo)
          for(l in 1:length(x)){
            if(x[l] <= med_x) {
              x_cat_2d[l] <- if(xo[l] <= med_x) 1 else 2
            } else {
              x_cat_2d[l] <- if(xo[l] <= med_x) 3 else 4
            }
          }
        }

        p.int <- chisq.test(x = x_cat_2d, y = Y[,(model$object$npar+1)])$p.value
        if(model$object$npar > 1){
          for(k in 2:model$object$npar){
            p <- chisq.test(x = x_cat_2d, y = Y[,(model$object$npar+k)])$p.value
            if(p < p.int) p.int <- p
          }
        }

        if(p.int < p.min) p.min <- p.int
      }
    }
  }

  ret <- c(p.min = p.min, p.curv = p.curv)

  return(ret)

}

}