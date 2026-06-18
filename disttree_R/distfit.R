###################################################################
# new version of distfit used as a fitting function in disttree #
###################################################################

distfit <- function(y, family = NO(), weights = NULL, start = NULL, start.eta = NULL,
                      vcov = TRUE, type.hessian =  c("checklist", "analytic", "numeric"),
                      method = "L-BFGS-B", estfun = TRUE, optim.control = list(), ...)
{

  ## FIX ME: error if parameters/eta are handed over in vector/matrix (only first values are chosen)
  ## start on par scale
  ## start.eta on link scale

  ## FIX ME: what to do if weights consists of zeros only

  type.hessian <- match.arg(type.hessian)

  ## match call
  cl <- match.call()

  ## number of observations
  ny <- NROW(y)

  ## check if all observations are equal
  allequ <- (length(unique(y)) == 1)


  ## check weights
  if(is.null(weights) || (length(weights)==0L)) weights <- as.vector(rep.int(1, ny))
  if(length(weights) != ny) stop("number of observations and length of weights are not equal")
  if(is.table(weights)) weights <- as.vector(weights)
  # check if all weights are 0 (happens for low number of trees, FIX ME: how to deal with this?)
  wzero <- all(weights == 0)
  if(wzero) {
    warning("all weights are 0")
    rval <- list(
      npar = NA,
      y = y,
      ny = ny,
      weights = weights,
      family = family,
      start = NA,
      starteta = NA,
      opt = NA,
      converged = NA,
      par = NA,
      eta = NA,
      hess = NA,
      vcov = NA,
      loglik = NA,
      call = cl,
      estfun = NA,
      method = NA,
      ddist = NA,
      pdist = NA,
      qdist = NA,
      rdist = NA
    )

    class(rval) <- "distfit"
    return(rval)
  }

  ## prepare family (done within distfamily()):
  # check format of the family input and if necessary transform it to the required familiy list
  # family input can be of one of the following formats:
  # - gamlss.family object
  # - gamlss.family function
  # - character string with the name of a gamlss.family object
  # - function generating a list with the required information about the distribution
  # - character string with the name of a function generating a list with the required information about the distribution
  # - list with the required information about the distribution
  # - character string with the name of a distribution for which a list generating function is provided in disttree

  if(!inherits(family, "disttree.family"))
    family <- distfamily(family)

  if(type.hessian == "checklist") {
    type.hessian <- if(is.null(family$hdist)) "numeric" else "analytic"
  }
  if(type.hessian == "numeric") family$hdist <- NULL
  if(type.hessian == "analytic" && is.null(family$hdist))
    stop("analytic calculation of hessian matrix not possible without list element hdist")


  # if family was handed over as a gamlss.dist family object and the distribution is censored:
  # data has to be converted to a Survival object (necessary family arguments: censtype and censpoint)
  if(family$censored) {
    if(family$gamlssobj) {
      if(family$censtype == "none" || is.null(family$censpoint)) stop("for censored gamlss.dist family objects the censoring point(s) has/have to be set (family argument censpoint)")
      if(!survival::is.Surv(y)){
        if(family$censtype == "left") y <- survival::Surv(y, y > family$censpoint, type = "left")
        if(family$censtype == "right") y <- survival::Surv(y, y < family$censpoint, type = "right")
        ## FIX ME: interval censored
        #if(family$censtype == "interval") y <- survival::Surv(y, ((y > family$censpoint[1]) * (y < family$censpoint[2])), type = "interval")
      }
    } else {
      if(survival::is.Surv(y)) {
        yc <- y
        y <- y[,1]
      }
    }
  }


  # FIXME:
  # the input argument fixed can be a character string or a list of character strings with the name(s) of the parameter(s) which are fixed
  # in this case their values must be set in the argument fixed.values

  # 2 families include fixed parameter(s): LNO() ... nu is fixed
  #                                        NET() ... nu and tau are fixed
  # any(family$parameters == FALSE)



  ## store y and select observations with weight > 0
  #FIXME# y.store <- y
  #FIXME# y <- y[weights > 0]

  ## number of observations = sum of weights (i.e., case weights)
  ## FIXME ## also need proportionality weights, i.e., weights = sum(weights > 0) ?
  # nobs <- sum(weights)

  ## notation:
  # par ... distribution parameters (mu, sigma, nu, tau)
  # eta ... coefficients of the linear predictor, here: intercept (g1(mu)=eta[1], g2(sigma)=eta[2], g3(nu)=eta[3], g4(tau)=eta[4])

  # if(np > 0L) m <- family$mu.linkinv(eta[1L])          # m ... mu           eta[1] ... g1(mu)        g1 ... link function
  # if(np > 1L) s <- family$sigma.linkinv(eta[2L])       # s ... sigma        eta[2] ... g2(sigma)     g2 ... link function
  # if(np > 2L) v <- family$nu.linkinv(eta[3L])          # v ... nu           eta[3] ... g3(nu)        g3 ... link function
  # if(np > 3L) t <- family$tau.linkinv(eta[4L])         # t ... tau          eta[4] ... g4(tau)       g4 ... link function



  ## set up negative log-likelihood
  nll <- function(eta) {

    #rval <- family$ddist(y, eta, log = TRUE, weights = weights, sum = FALSE)

    rval <- suppressWarnings(try(family$ddist(y, eta, log = TRUE, weights = weights, sum = TRUE), silent = TRUE))
    if(inherits(rval, "try-error")) {
      print("try-error in ddist")
      return(1.7e+300)
    } else {
      if(any(is.na(rval))) {
        print("NAs in ddist")
        return(1.7e+300)
      } else {
        if(any(abs(rval) == Inf)) {
          print(c("Infinity in ddist"))
          return(1.7e+300)
        } else {
          nloglik <- - rval
          if(abs(nloglik) == Inf) return(1.7e+300)
          return(nloglik)
        }
      }
    }
  }


  ## set up gradient
  grad <- function(eta) {
    gr <- - family$sdist(y, eta, weights = weights, sum = TRUE)
    if(any(is.na(gr))) {
      gr[is.na(gr)] =  1.7e+300
      print(c(eta,"gradient = NaN"))
    } else {
      if(any(gr == -Inf)) {
        gr[gr==-Inf] = -1.7e+300
        print(c(eta,"gradient = -Inf"))
      }
      if(any(gr ==  Inf)) {
        gr[gr== Inf] =  1.7e+300
        print(c(eta,"gradient = Inf"))
        print(length(y))
      }
    }
    return(gr)
  }

  ## calculate initial values if necessary or otherwise transform initial values for the distribution parameters to initial values on the link scale
  if((is.null(start) && is.null(start.eta)) | family$mle){
    if(NROW(y)>1 & !allequ) {
      starteta <- family$startfun(y, weights = weights)
    } else {
      ## FIX ME: replacements of starting values apart from location
      if(NROW(y)==1 | allequ){
        starteta <- try(family$startfun(y, weights = weights))
        if(inherits(starteta, "try-error") | any(is.na(starteta))){
          starteta <- family$linkfun(c(unique(y), rep.int(1e-10, length(family$link)-1)))  ## FIX ME: set all other parameters to 1e-10?
        }
        warning("only one observation or only equal observations in distfit")
      } else warning("no observation in distfit")
    }

    #all0 <- if(is.Surv(y)) all(y[,2]==0) else all(y==0)
    if(any(starteta[(family$link == "log" | family$link == "logit")] == -Inf)){
      starteta[which((family$link == "log" | family$link == "logit") & (starteta==-Inf))] <- log(0.0001)
      if(allequ) print("one node with all equal observations: sigma set to 0.0001") else
        print("parameter on link scale = -Inf: parameter set to 0.0001")
    }
  } else {
    if(!(is.null(start))){
      # check that those parameters with a log-function in the linkfun are not negative or zero
      # (if they are, replace them by 1e-10)
      start[start[family$link == "log" | family$link == "logit"]<=0] <- 1e-10
      starteta <- family$linkfun(start)
    }
    if(!(is.null(start.eta))){
      starteta <- start.eta
    }
    if(any(is.na(starteta))) {
      warning("NAs in starteta")
      #print("y=")
      #print(y)
      #print("weights=")
      #print(weights)
      #print("start=")
      #print(start)
    }
  }

  if(!family$mle) {
    ## optimize negative log-likelihood
    opt <- try(optim(par = starteta, fn = nll, gr = grad, method = method,
                     hessian = (type.hessian == "numeric"), control = optim.control, ...), silent = TRUE)
    if(inherits(opt, "try-error")) {
      if(method != "L-BFGS-B") method <- "L-BFGS-B"
      warning("Error in 'optim()' for given method and additional arguments in '...',
              optimization restarted with 'L-BFGS-B' and additional arguments ignored")
      opt <- try(optim(par = starteta, fn = nll, gr = grad, method = method,
                       hessian = (type.hessian == "numeric"), control = optim.control), silent = TRUE)
      if(inherits(opt, "try-error")) {
        warning("Error in 'optim()' for method 'L-BFGS-B',
                optimization restarted with 'BFGS' and additional arguments ignored")
        ## FIX ME: first keep additional arguments, only if this fails as well change method
        ## FIX ME: order of steps?
        method <- "BFGS"
        opt <- try(optim(par = starteta, fn = nll, gr = grad, method = method,
                         hessian = (type.hessian == "numeric"), control = optim.control), silent = TRUE)
        if(inherits(opt, "try-error")) {
          warning("Error in 'optim()' for method 'L-BFGS-B',
                optimization restarted with 'Nelder-Mead' and additional arguments ignored")
          #print(starteta)
          method <- "Nelder-Mead"
          opt <- optim(par = starteta, fn = nll, method = method,
                       hessian = (type.hessian == "numeric"), control = optim.control)
        }
      }
    }
    ## extract parameters
    eta <- opt$par
    names(eta) <- names(starteta)
    par <- family$linkinv(eta)

    ## loglikelihood value
    loglik = -opt$value

    converged <- (opt$convergence == 0)   # optim returns 0 for successful completion

  } else {
    eta <- starteta
    par <- family$linkinv(eta)
    loglik <- family$ddist(y, eta, log = TRUE, weights = weights, sum = TRUE)
    opt <- NULL
    converged <- TRUE
  }


  ## hess matrix (second-order partial derivatives of the (positive) log-likelihood function) and Hessian estimate for the variance-covariance matrix
  if(vcov) {
    if(is.null(family$hdist)) {
      if(family$mle) {
        nhess <- try(optim(par = eta, fn = nll, gr = grad, method = "L-BFGS-B",
                         hessian = TRUE, control = optim.control, ...)$hessian)
        if(inherits(nhess, "try-error")) {
          #print("opt try-error in hess with gr=grad, L-BFGS-B")
          nhess <- try(optim(par = eta, fn = nll, gr = grad, method = "BFGS",
                       hessian = TRUE, control = optim.control, ...)$hessian)
          if(inherits(nhess, "try-error")) {
            #print("opt try-error in hess with gr=grad, BFGS")
            nhess <- optim(par = eta, fn = nll, method = "BFGS",
                               hessian = TRUE, control = optim.control, ...)$hessian
          }
        }
        hess <- -nhess
      } else {
        hess <- -opt$hessian
      }
    } else {
      hess <- family$hdist(y, eta, weights = weights)
    }
    hess <- as.matrix(hess)
    colnames(hess) <- rownames(hess) <- names(eta)


    #vcov for link coefficients eta
    vc <- try(solve(-hess), silent = TRUE)
    if(inherits(vc, "try-error")) {
      vc <- try(qr.solve(-hess), silent = TRUE)
      if(inherits(vc, "try-error")) {
        vc <- try(chol2inv(chol(-hess)))
        if(inherits(vc, "try-error")) {
          print(-hess)
          print("hessian matrix is 'numerically' singular")
        }
      }
    }


    # if(inherits(vc, "try-error")) {
    #   vc <- MASS::ginv(-hess)
    # }

    vc <- as.matrix(vc)
    colnames(vc) <- rownames(vc) <- colnames(hess)

  } else {
    hess <- NULL
    vc <- NULL
  }


  ## estfun
  # each column represents one distribution parameter (1.col -> dldm * dmdpar = "dldeta.mu", 2.col -> dldd * dddpar = "dldeta.sigma", ...)
  # (first-order partial derivatives of the (positive) log-likelihood function)
  if(estfun) {
    ef <- if(allequ) {
      matrix(0, ncol = length(eta), nrow = ny)
    } else {
    # estfun for link coefficients eta
    weights * family$sdist(y, eta, sum = FALSE)   ## FIX ME: cut out rows with weight = 0? -> No! index is of importance for independence tests (relation to covariates)
    }
  } else {
    ef <- NULL
  }


  ##### additional functions with set parameters

  ## density function
  if(family$gamlssobj && family$censored){
    ddist <- function(x, log = FALSE) {
      if(!survival::is.Surv(x)){
        if(family$censtype == "left") eval <- family$ddist(survival::Surv(x, x > family$censpoint, type = "left"), eta = eta, log = log)
        if(family$censtype == "right") eval <- family$ddist(survival::Surv(x, x < family$censpoint, type = "right"), eta = eta, log = log)
        ## FIX ME: interval censored
      } else eval <- family$ddist(x, eta = eta,  log=log)
      return(eval)
    }
  } else ddist <- function(x, log = FALSE) family$ddist(x, eta = eta, log = log)

  ## cumulative distribution function
  pdist <- function(q, lower.tail = TRUE, log.p = FALSE) family$pdist(q, eta = eta, lower.tail = lower.tail, log.p = log.p)

  ## quantile function
  qdist <- function(p, lower.tail = TRUE, log.p = FALSE) family$qdist(p, eta = eta, lower.tail = lower.tail, log.p = log.p)

  ## random function
  rdist <- if(is.null(family$rdist)) NULL else function(n) family$rdist(n, eta = eta)



  ## return value
  rval <- list(
    npar = length(par),
    y = y,
    ny = ny,
    weights = weights,
    family = family,
    start = start,
    starteta = starteta,
    opt = opt,
    converged = converged,
    par = par,
    eta = eta,
    hess = hess,
    vcov = vc,
    loglik = loglik,
    call = cl,
    estfun = ef,
    method = method,
    ddist = ddist,
    pdist = pdist,
    qdist = qdist,
    rdist = rdist
  )

  class(rval) <- "distfit"
  return(rval)
}





## methods
nobs.distfit <- function(object, ...) return(object$ny)

coef.distfit <- function(object, type = c("parameter", "link"), ...) {
  type <- match.arg(type)
  if(type == "link") return(object$eta)
  if(type == "parameter") return(object$par)
}

get_expectedvalue <- function(object, par) {

  if(is.vector(par)) {
    # 1-parametric distribution
    if(length(object$info$family$link) == 1){
      par <- as.matrix(par)
    } else {
    # only 1 observation
      par <- t(as.matrix(par))
    }
  }

  if(inherits(object, "distfit")) {
    censored <- object$family$censored
    family.name <- object$family$family.name
    expectedvalue <- ifelse(is.null(object$family$expectedvalue), FALSE, object$family$expectedvalue)
  }
  if(inherits(object, "disttree")) {
    censored <- object$info$family$censored
    family.name <- object$info$family$family.name
    expectedvalue <- ifelse(is.null(object$info$family$expectedvalue), FALSE,
      object$info$family$expectedvalue)
  }

  if(expectedvalue & (censored | "truncated" %in% strsplit(family.name, " ")[[1]]))
    warning("parameter 'expectedvalue' of family is ignored, as family is censored/truncated...")

  ## FIXME: replace censored/truncated case with crch implementation
  ## (here only for left censored at 0 -> TO DO: general form)
  ## other censored distributions currently not possible (other than normal or logistic)
  ## (general implementation for closed form solutions available?)
  if(censored) {
    if("Normal" %in% strsplit(family.name, " ")[[1]]){
      mu <- par[,1]
      sigma <- par[,2]
      expv <- pnorm(mu/sigma) * (mu + sigma * (dnorm(mu/sigma) / pnorm(mu/sigma)))
    } else if("Logistic" %in% strsplit(family.name, " ")[[1]]){
        location <- par[,1]
        scale <- par[,2]
        expv <- (1 - (1 / (1 + exp(location/scale)))) * scale * (1 + exp(-location/scale)) * log(1 + exp(location/scale))
    } else {
      ## FIX ME: expected value for other censored distributions:
      warning("For censored distributions other than the censored normal and censored logistic distribution
              the location parameter is returned as response/expected value.")
      expv <- par[,1]
    }
    return(expv)
  }

  if("truncated" %in% strsplit(family.name, " ")[[1]]) {
    if("Normal" %in% strsplit(family.name, " ")[[1]]){
      mu <- par[,1]
      sigma <- par[,2]
      expv <- (mu + sigma * (dnorm(mu/sigma) / pnorm(mu/sigma)))
    } else {
      ## FIX ME: expected value for other truncated distributions:
      warning("For truncated distributions other than the truncated normal distribution
                the location parameter is returned as response/expected value.")
      expv <- par[,1]
    }
    return(expv)
  }

  if(expectedvalue){
    expv <- par[,1]
    return(expv)
  }


  # integrate over function f = x * density function with estimated parameters plugged in (already in the object for class 'distfit')
  if(inherits(object, "distfit")) {
    f <- function(x) {
      dens <- try(object$ddist(x, log = FALSE), silent = TRUE)
      # if function is only defined on a limited range
      if(inherits(dens, "try-error")) dens <- 0
      x * dens
    }
    expv <- try(integrate(f,-Inf, Inf), silent = TRUE)
    if(inherits(expv, "try-error")) {
      expv <- try(integrate(f,-Inf, Inf, rel.tol = 1e-03))
      if(inherits(expv, "try-error")) {
        expv <- try(integrate(f,-Inf, Inf, rel.tol = 1e-02))
        if(inherits(expv, "try-error")) {
          print("rel.tol had to be set to 0.1 to calculated expected values for predictions")
          #print(coef(object))
          expv <- integrate(f,-Inf, Inf, rel.tol = 1e-01)
        }
      }
    }
    expv <- expv[[1]]
  }

  if(inherits(object, "disttree")){
    expv <- numeric(length = NROW(par))
    for(i in 1:NROW(par)){
      eta <- unlist(object$info$family$linkfun(par[i,]))
      f <- function(x) {
        dens <- try(object$info$family$ddist(x, eta = eta, log = FALSE), silent = TRUE)
        # if function is only defined on a limited range
        if(inherits(dens, "try-error")) dens <- 0
        x * dens
      }
      val <- try(integrate(f,-Inf, Inf), silent = TRUE)
      if(inherits(expv, "try-error")) {
        val <- try(integrate(f,-Inf, Inf, rel.tol = 1e-03))
        if(inherits(expv, "try-error")) {
          val <- try(integrate(f,-Inf, Inf, rel.tol = 1e-02))
          if(inherits(expv, "try-error")) {
            print("rel.tol had to be set to 0.1 to calculated expected values for predictions")
            #print(coef(object))
            val <- integrate(f,-Inf, Inf, rel.tol = 1e-01)
          }
        }
      }
      expv[i] <- val[[1]]
    }
  }

  return(expv)
}


predict.distfit <- function(object, type = c("parameter", "response"), ...) {
  # for type = "response": calculation of the expected value
  # of the given distribution with the calculated parameters

  # per default 'type' is set to 'parameter'
  type <- match.arg(type)

  if(type == "response"){
    return(get_expectedvalue(object, object$par))
  }  else {
    return(object$par)
  }
}

vcov.distfit <- function(object, type = c("parameter", "link"), ...) {
  type <- match.arg(type)
  if(type == "link") return(object$vcov)
  if(type == "parameter"){
    ## delta method
    delta.m <- diag(object$family$linkinvdr(object$eta))
    colnames(delta.m) <- rownames(delta.m) <- names(object$par)
    return(delta.m %*% object$vcov %*% delta.m)
  }
}

estfun.distfit <- function(x, ...) return(x$estfun)

logLik.distfit <- function(object, ...) structure(object$loglik, df = object$npar, class = "logLik")

bread.distfit <- function(x, type = c("parameter", "link"), ...) {
  type <- match.arg(type)
  if(type == "parameter") return(vcov(x, type = "parameter") * x$ny)
  if(type == "link") return(x$vcov * x$ny)
}

confint.distfit <- function(object, parm, level = 0.95, type = c("parameter", "link"), ...) {
  type <- match.arg(type)
  np <- object$npar

  if(type == "link"){
    coef <- object$eta
    vcov <- object$vcov
    # FIX ME: vcov on link scale: values around zero might be negative => error using sqrt
  }
  if(type == "parameter"){
    coef <- object$par
    vcov <- vcov(object, type = "parameter")
  }

  left <- (1-level)/2
  right <- 1-left

  if(missing(parm)){
    use.parm <- rep(TRUE,length = np)
  } else {
    use.parm <- logical(length = np)
    if(("mu" %in% parm) || (paste0(object$family$link[1],"(mu)") %in% parm) || 1 %in% parm) use.parm[1] <- TRUE
    if(np > 1L) {
      if(("sigma" %in% parm) || (paste0(object$family$link[2],"(sigma)") %in% parm) || 2 %in% parm) use.parm[2] <- TRUE
      if(np > 2L) {
        if(("nu" %in% parm) || (paste0(object$family$link[3],"(nu)") %in% parm) || 3 %in% parm) use.parm[3] <- TRUE
        if(np > 3L) if(("tau" %in% parm) || (paste0(object$family$link[4],"(tau)") %in% parm) || 4 %in% parm) use.parm[4] <- TRUE
      }
    }
  }

  confint <- NULL
  if(use.parm[1]) {
    confint1 <- c(coef[1] + qnorm(left) * sqrt(vcov[1,1]), coef[1] + qnorm(right) * sqrt(vcov[1,1]))
    confint <- rbind(confint, confint1)
  }
  if(np > 1L) {
    if(use.parm[2]) {
      confint2 <- c(coef[2] + qnorm(left) * sqrt(vcov[2,2]), coef[2] + qnorm(right) * sqrt(vcov[2,2]))
      confint <- rbind(confint, confint2)
    }
    if(np > 2L) {
      if(use.parm[3]) {
        confint3 <- c(coef[3] + qnorm(left) * sqrt(vcov[3,3]), coef[3] + qnorm(right) * sqrt(vcov[3,3]))
        confint <- rbind(confint, confint3)
      }
      if(np > 3L) {
        if(use.parm[4]) {
          confint4 <- c(coef[4] + qnorm(left) * sqrt(vcov[4,4]), coef[4] + qnorm(right) * sqrt(vcov[4,4]))
          confint <- rbind(confint, confint4)
        }
      }
    }
  }


  confint <- as.matrix(confint)
  colnames(confint) <- c(paste0(left," %"), paste0(right," %"))
  rownames(confint) <- names(coef[use.parm])

  confint
}

print.distfit <- function(x, digits = max(3, getOption("digits") - 3), ...)
{
  cat("Fitted distributional model (",x$family$family.name,")\n\n", sep = "")
  if(!(x$family$mle) && !x$converged) {
    cat("Model did not converge\n")
  } else {
    if(length(x$par)) {
      cat("Distribution parameter(s):\n")
      print.default(format(x$par, digits = digits), print.gap = 2, quote = FALSE)
      cat("\n")
    } else {
      cat("No parameters \n\n")
    }
    cat(paste("Log-likelihood: ", format(x$loglik, digits = digits), "\n", sep = ""))
    if(length(x$npar)) {
      cat(paste("Df: ", format(x$npar, digits = digits), "\n", sep = ""))
    }
    cat("\n")
  }

  invisible(x)
}

summary.distfit <- function(object, ...)
{
  ## residuals
  object$residuals <- object$y - predict.distfit(object, type = "response")
  if(length(object$par)>1) {
    object$residuals <- object$residuals / object$par[2]
  } else {warning("one-parametric distribution, residuals are not standardized")}

  ## extend coefficient table
  cf <- as.vector(object$par)
  if (!is.null(object$vcov)){
    se <- sqrt(diag(vcov(object, type = "parameter")))
    cf <- cbind(cf, se)
    colnames(cf) <- c("Estimate", "Std. Error")
  } else{
    cf <- matrix(cf, ncol = 1)
    colnames(cf) <- c("Estimate")
  }
  rownames(cf) <- names(object$par)
  object$coefficients <- cf

  ## return
  class(object) <- "summary.distfit"
  object
}


print.summary.distfit <- function(x, digits = max(3, getOption("digits") - 3), ...)
{
  cat("\nCall:", deparse(x$call, width.cutoff = floor(getOption("width") * 0.85)), "", sep = "\n")

  if(!(x$family$mle) && !x$converged) {
    cat("model did not converge\n")
  } else {
    cat(paste("Distribution Family:\n", x$family$family.name, "\n\n", sep = ""))

    cat(paste("Standardized residuals:\n", sep = ""))
    print(structure(round(as.vector(quantile(x$residuals)), digits = digits),
                    .Names = c("Min", "1Q", "Median", "3Q", "Max")))

    if(NROW(x$coefficients)) {
      cat(paste("\nDistribution parameter(s):\n", sep = ""))
      printCoefmat(as.data.frame(x$coefficients), digits = digits, signif.legend = FALSE)
    } else cat("\nNo parameters\n")

    cat("\nLog-likelihood:", formatC(x$loglik, digits = digits), "on", x$npar, "Df\n")
    if(!x$family$mle)
      cat(paste("Number of iterations in", x$method, "optimization:\n",
                "function:", x$opt$counts[1L], "\n",
                "gradient:", x$opt$counts[2L]))
  }

  invisible(x)
}


getSummary.distfit <- function(object, alpha = 0.05, ...) {
  ## extract coefficient summary
  s <- summary(object)
  cf <- s$coefficients
  ## augment with confidence intervals
  cval <- qnorm(1 - alpha/2)
  cf <- cbind(cf,
              cf[, 1] - cval * cf[, 2],
              cf[, 1] + cval * cf[, 2])

  colnames(cf) <- c("est", "se", "lwr", "upr")

  ## return everything
  return(list(
    family = object$family$family.name,
    coef = cf,
    sumstat = c(
      "N" = object$nobs,
      "logLik" = as.vector(logLik(object)),
      "AIC" = AIC(object),
      "BIC" = AIC(object, k = log(object$ny))
    ),
    call = object$call
  ))
}


# FIXME: check correct calculations for each available type
residuals.distfit <- function(object, type = c("standardized", "pearson", "response"), ...) {
  if(match.arg(type) == "response") {
    object$y - predict.distfit(object, type = "response")
  } else {
    if(length(object$par)>1) {
      (object$y - predict.distfit(object, type = "response")) / object$par[2]
    } else {stop("one-parametric distribution, residuals are not standardized")}
  }
}



## FIXME: revise plotting options
plot.distfit <- function(x,
                         main = "", xlab = "", ylab = "Density",
                         fill = "lightgray", col = "darkred", lwd = 1.5,
                         ...)
{
  ## FIX ME: barplot instead of hist for discrete distributions
  if(isTRUE(all.equal(x$y, round(x$y)))) {

    ## barplot instead of hist:
    #ytab <- table(x$y)
    #values <- as.numeric(names(ytab))
    #absfreq <- as.numeric(ytab)
    #relfreq <- absfreq / sum(absfreq)
    #relytab <- ytab / sum(absfreq)
    #bars <- barplot(relytab, xlim = c(min(values), max(values)), ylim = c(-0.1, 1.1),
    #                xlab = xlab , ylab = ylab)
    #abline(h = 0)
    #lines(values*1.2 + 0.7, x$ddist(values), type = "b", lwd = 2, pch = 19, col = 2)

    # using hist:
    histogram <- c(
      list(x = x$y, main = main, xlab = xlab, ylab = ylab, col = fill),
      list(...)
    )
    histogram$freq <- FALSE
    histogram$probability <- TRUE
    histogram$breaks <- seq(from = min(x$y), to = max(x$y) + 1) - 0.5
    # histogram$breaks <- seq(from = min(x$y), to = 2*max(x$y) + 1)/2 - 0.25
    histogram <- do.call("hist", histogram)
    yrange <- seq(from = min(x$y), to = max(x$y))
    lines(yrange, x$ddist(yrange), col = col, lwd = lwd)

  } else {

    histogram <- c(
      list(x = x$y, main = main, xlab = xlab, ylab = ylab, col = fill),
      list(...)
    )
    histogram$freq <- FALSE
    histogram$probability <- TRUE
    #histogram$breaks <- seq(from = min(x$y), to = max(x$y) + 1) - 0.5
    histogram <- do.call("hist", histogram)
    yrange <- seq(from = histogram$breaks[1L],
                  to = histogram$breaks[length(histogram$breaks)],
                  length.out = 100L)
    lines(yrange, x$ddist(yrange), col = col, lwd = lwd)
  }
}





## FIXME: summary (also on link scale?)
#summary.distfit <- function (object, type = "parameter", ...){
#  if(type == "link"){
#    coef <- object$eta
#    vcov <- object$vcov
#  }
#  if(type == "parameter"){
#    coef <- object$par
#    vcov <- vcov(object, type = "parameter")
#  }
#
#  se <- sqrt(diag(vcov))
#  TAB <- cbind(Estimate = coef,
#               StdErr = se)
#
#  sumlist <- list(Call = object$call,
#                  Family = object$family,
#                  Parameter = TAB,
#                  Estimated_covariance_matrix = vcov,
#                  LogLikelihood = object$loglik
#                  # LogLikelihood = logLik(object)
#  )
#  class(sumlist) <- "summary.distfit"
#  sumlist
#}