#### distfamily
# generates a list of class disttree.family with all the information required in distfit/-tree/-forest
# first checks format of the family input and if necessary transforms it to the required family list
# family input can be of one of the following formats:
# - gamlss.family object
# - gamlss.family function
# - character string with the name of a gamlss.family object
# - function generating a list with the required information about the distribution
# - character string with the name of a function generating a list with the required information about the distribution
# - list with the required information about the distribution
# - character string with the name of a distribution for which a list generating function is provided in disttree
# other input arguments: bd (binomial denominator), censpoint (for censored gamlss.family objects)

distfamily <- function(family, bd = NULL, censpoint = NULL){

  if(is.character(family)) {
    getfamily <- try(getAnywhere(paste("dist", family, sep = "_")), silent = TRUE)
    if(length(getfamily$objs) == 0L) getfamily <- try(getAnywhere(family), silent = TRUE)
    if(length(getfamily$objs) == 0L) {
      stop("unknown 'family' specification")
    } else {
      gamlssobj <- ("gamlss.dist" %in% unlist(strsplit(getfamily$where[1], split = ":")))
      family <- getfamily[[2]][[1]]() #first found is chosen
      family$gamlssobj <- gamlssobj
    }
    #if(!(inherits(family, "try-error")))family <- family[[2]]$`package:disttree`()
    # FIX ME: better selection of dist function
  }

  # if family is a gamlss family object or gamlss family function
  if(is.function(family))
    family <- family()
  if(inherits(family, "gamlss.family"))
    family <- distfamily_gamlss(family, bd = bd, censpoint = censpoint)

  if(!is.list(family)) stop ("unknown family specification")
  if(!(all(c("ddist", "sdist", "link", "linkfun", "linkinv", "mle", "startfun") %in% names(family))))
    stop("family needs to specify a list with ddist, sdist, link, linkfun, linkinv, mle and startfun")
  # linkinvdr only used in the method vcov for type = "parameter"

  class(family) <- "disttree.family"
  return(family)
}







##### distfamily_gamlss
# is called in distfamily if family is a gamlss.family object
# input: gamlss.dist family object and optional arguments bd and censpoint


distfamily_gamlss <- function(family, bd = NULL, censpoint = NULL)
{
  np <- sum(family$parameter == TRUE)

  gamlss_name <- unlist(strsplit(family$family[2], split = " "))
  censored <- ("censored" %in% gamlss_name)
  if(!censored & !is.null(censpoint))
    stop("censpoint can only be defined for a censored family")
  censtype <- if(censored) gamlss_name[1] else "none"

  ## list of families which require an additional parameter bd (binomial denominator)
  # by default bd is set to to 10 for BB() and to 1 for all the others
  .distfit.bi.list <- c("BI", "Binomial", "BB", "Beta Binomial", "ZIBI", "ZIBB", "ZABI", "ZABB") # binomial denominators
  if(any(family$family%in%.distfit.bi.list) && is.null(bd)) ifelse(family$family[1] == "BB", bd <- 10, bd <- 1)


  ## families which have fixed parameters: LNO (log normal (Box-Cox)), NET
  # define fixed parameters locally within distfamily_gamlss, such that the fixed parameters don't appear in the output list anymore
  # for LNO a value nu.start is needed for the evaluation of mu.initial
  if(family$family[1] == "LNO") nu.start <- NULL

  if(FALSE) {
    if(family$nopar != np){
      if(family$family[1] == "LNO") nu <- nu.start <- 0
      if(family$family[1] == "NET") {
        nu <- 1.5
        tau <- 2
      }

      #general form: (problem: y required in .initial)
      #if(!family$parameters$mu) {eval(family$mu.initial)}
      #if(!family$parameters$sigma) {eval(family$sigma.initial)}
      #if(!family$parameters$nu) {eval(family$nu.initial)}
      #if(!family$parameters$tau) {eval(family$tau.initial)}
    }
  }


  ## notation:
  # par ... distribution parameters (mu, sigma, nu, tau)
  # eta ... coefficients of the linear predictor,
  #         here: intercept (g1(mu)=eta[1], g2(sigma)=eta[2], g3(nu)=eta[3], g4(tau)=eta[4])

  # m ... mu           eta[1] ... g1(mu)        g1 ... link function
  # d ... sigma        eta[2] ... g2(sigma)     g2 ... link function
  # v ... nu           eta[3] ... g3(nu)        g3 ... link function
  # t ... tau          eta[4] ... g4(tau)       g4 ... link function



  ## Define all necessary functions depending on the number of parameters

  # get derivative function as required in the list (input: gamlss.dist derivative function)
  getderivfun <- function(fun){
    arg <- names(formals(fun))
    par.id <- as.numeric(na.omit(match(arg, c("mu", "sigma", "nu", "tau"))))
    if("y" %in% arg) {
      if("bd" %in% arg) {
        derivfun <- function(y, par) {
          input <- list()
          input$y <- y
          input <- c(input, par[par.id])
          input$bd <- bd
          val <- do.call(fun, input)
          if(length(val) == 1L) val <- rep(val, NROW(y))
          return(val)
        }
      } else {
        derivfun <- function(y, par) {
          input <- list()
          input$y <- y
          input <- c(input, par[par.id])
          val <- do.call(fun, input)
          if(length(val) == 1L) val <- rep(val, NROW(y))
          return(val)
        }
      }
    } else {
      if("bd" %in% arg) {
        derivfun <- function(y, par) {
          input <- as.list(par[par.id])
          input$bd <- bd
          return(rep(do.call(fun, input), NROW(y)))
        }
      } else {
        derivfun <- function(y, par) {
          input <- as.list(par[par.id])
          return(rep(do.call(fun, input), NROW(y)))
        }
      }
    }
    return(derivfun)
  }


  ## define inner and outer derivative functions

  if(np > 0L){

    # define names
    parnames <- c("mu")
    linknames <- c(family$mu.link)
    if(family$mu.link == "identity") etanames <- c("mu") else etanames <- paste0(family$mu.link, "(mu)")


    # inner derivative functions (dmdeta, d2mdeta2)
    dmdeta <- function(eta) return(family$mu.dr(eta[1]))
    if(family$mu.link=="identity") d2mdeta2 <- function(eta) return(0)
    if(family$mu.link=="log")      d2mdeta2 <- function(eta) return(exp(eta[1]))
    if(family$mu.link=="logit")    d2mdeta2 <- function(eta) return(exp(eta[1]) * (1-exp(eta[1])) / ((1+exp(eta[1]))^3))

    # outer derivative functions (dldm, d2ldm2)
    dldm <- getderivfun(family$dldm)
    d2ldm2 <- getderivfun(family$d2ldm2)

    ## FIX ME: alternative version, pro: shorter, contra: if-conditions inside the function
    if(FALSE){
      arg.dldm <- names(formals(family$dldm))
      par.id.dldm <- getpar(family$dldm)
      dldm <- function(y, par){
        input <- list()
        if("y" %in% arg.dldm) input$y <- y
        input <- c(input, par[par.id.dldm])
        if("bd" %in% arg.dldm) input$bd <- bd
        return(do.call(family$dldm, input))
      }
    }

  }


  if(np > 1L){

    # define names
    parnames <- c(parnames,"sigma")
    linknames <- c(linknames, family$sigma.link)
    if(family$sigma.link == "identity") etanames <- c(etanames,"sigma") else etanames <- c(etanames, paste0(family$sigma.link, "(sigma)"))

    # inner derivative functions (dddeta, d2ddeta2)
    dddeta <- function(eta) return(family$sigma.dr(eta[2]))
    if(family$sigma.link=="identity") d2ddeta2 <- function(eta) return(0)
    if(family$sigma.link=="log")      d2ddeta2 <- function(eta) return(exp(eta[2]))
    if(family$sigma.link=="logit")    d2ddeta2 <- function(eta) return(exp(eta[2]) * (1-exp(eta[2])) / ((1+exp(eta[2]))^3))

    # outer derivative functions (dldd, d2ldd2, d2ldmdd)
    dldd <- getderivfun(family$dldd)
    d2ldd2 <- getderivfun(family$d2ldd2)
    d2ldmdd <- getderivfun(family$d2ldmdd)

  }


  if(np > 2L){

    # define names
    parnames <- c(parnames,"nu")
    linknames <- c(linknames, family$nu.link)
    if(family$nu.link == "identity") etanames <- c(etanames,"nu") else etanames <- c(etanames, paste0(family$nu.link, "(nu)"))

    # inner derivative functions (dvdeta, d2vdeta2)
    dvdeta <- function(eta) return(family$nu.dr(eta[3]))
    if(family$nu.link=="identity") d2vdeta2 <- function(eta) return(0)
    if(family$nu.link=="log")      d2vdeta2 <- function(eta) return(exp(eta[3]))
    if(family$nu.link=="logit")    d2vdeta2 <- function(eta) return(exp(eta[3]) * (1-exp(eta[3])) / ((1+exp(eta[3]))^3))

    # outer derivatives (dldv, d2ldv2, d2ldmdv, d2ldddv)
    dldv <- getderivfun(family$dldv)
    d2ldv2 <- getderivfun(family$d2ldv2)
    d2ldmdv <- getderivfun(family$d2ldmdv)
    d2ldddv <- getderivfun(family$d2ldddv)
  }


  if(np > 3L){

    # define names
    parnames <- c(parnames,"tau")
    linknames <- c(linknames, family$tau.link)
    if(family$tau.link == "identity") etanames <- c(etanames,"tau") else etanames <- c(etanames, paste0(family$tau.link, "(tau)"))

    # note: in this case/section no adaption for families of the list .distfit.bi.list since none of them includes the 4th parameter tau

    # inner derivatives (dtdeta, d2tdeta2)
    dtdeta <- function(eta) return(family$tau.dr(eta[4]))
    if(family$tau.link=="identity")  d2tdeta2 <- function(eta) return(0)
    if(family$tau.link=="log")       d2tdeta2 <- function(eta) return(exp(eta[4]))
    if(family$tau.link=="logit")     d2tdeta2 <- function(eta) return(exp(eta[4]) * (1-exp(eta[4])) / ((1+exp(eta[4]))^3))

    # outer derivatives (dldt, d2ldt2, d2ldmdt, d2ldddt, d2ldvdt)
    dldt <- getderivfun(family$dldt)
    d2ldt2 <- getderivfun(family$d2ldt2)
    d2ldmdt <- getderivfun(family$d2ldmdt)
    d2ldddt <- getderivfun(family$d2ldddt)
    d2ldvdt <- getderivfun(family$d2ldvdt)

  }



  ## define startfunction, complete derivative functions dpardeta, d2pardeta2, dldpar, d2ldpar2 according to the number of parameters

  ## TO DO: change starting expressions: eg. mean -> weighted.mean (use gsub to substitute function names)
  weight_mean_expression <- if(censored) {
    function(e) {
      e <- gsub("mean(y[, 1])", "weighted.mean(y[, 1], weights)", e, fixed = TRUE)
      e <- parse(text = e)
      return(e)
    }
  } else {
    function(e) {
      e <- gsub("mean(y)", "weighted.mean(y, weights)", e, fixed = TRUE)
      e <- parse(text = e)
      return(e)
    }
  }


  if(np == 1L){

    # define function for the calculation of initial values on the link scale
    startfun <- function(y, weights = NULL) {
      mu <- NULL
      if(is.null(weights) || (length(weights)==0L)) {
        eval(family$mu.initial)
        starteta <- c(family$mu.linkfun(mean(mu)))
      } else {
        eval(weight_mean_expression(family$mu.initial))
        starteta <- c(family$mu.linkfun(weighted.mean(mu, weights)))
      }
      names(starteta) <- etanames
      return(starteta)
    }

    # define link function
    linkfun <- function(par){
      eta <- c(family$mu.linkfun(par[1]))
      names(eta) <- etanames
      return(eta)
    }

    # define function to get distribution parameters
    linkinv <- function(eta){
      par <- c(family$mu.linkinv(eta[1]))
      names(par) <- parnames
      return(par)
    }

    # define functions that return inner derivatives as vector / matrix:
    dpardeta <- function(eta){
      return(c(dmdeta(eta)))
    }

    d2pardeta2 <- function(eta){
      return(c(d2mdeta2(eta)))
    }


    # define functions that return outer derivatives as matrix / list of matrices:
    dldpar <- function(y, par){
      dmatrix <- cbind(dldm(y, par))
      return(dmatrix)
    }

    d2ldpar2 <- function(y, par){

      d2matrix <- rbind(cbind(d2ldm2(y, par)))

      # d2matrix is of size (1*ny x 1)
      # transform to a list of matrices (length of the list equals the number of observations ny)
      # for each observation a matrix of size (1x1) is stored in d2list

      d2list <- list()
      ny <- NROW(y)
      length(d2list) <- ny
      for(i in 1:ny){
        d2list[[i]] <- d2matrix[c(i),]
      }

      return(d2list)
    }


    # define p-, q- and r-function  (r-function not available for all GAMLSS families, e.g. rLOlc)
    pdist <- function(q, eta, lower.tail = TRUE, log.p = FALSE) {
      par <- linkinv(eta)
      do.call(get(paste0("p",family$family[1])),
              list(q, par[1],
                   lower.tail = lower.tail, log.p = log.p))
    }
    qdist <- function(p, eta, lower.tail = TRUE, log.p = FALSE) {
      par <- linkinv(eta)
      do.call(get(paste0("q",family$family[1])),
              list(p, par[1],
                   lower.tail = lower.tail, log.p = log.p))
    }
    rfun.available <- try(get(paste0("r",family$family[1])), silent = TRUE)
    if(inherits(rfun.available, "try-error")) {
      # warning("r-function not available for this family object")
      rdist <- NULL
    } else {
      rdist <- function(n, eta) {
        par <- linkinv(eta)
        do.call(get(paste0("r",family$family[1])), list(n, par[1]))
      }
    }
  }


  if(np == 2L){

    # define function for the calculation of initial values on the link scale
    startfun <- function(y, weights = NULL) {
      mu <- sigma <- NULL
      if(is.null(weights) || (length(weights)==0L)) {
        eval(family$mu.initial)
        eval(family$sigma.initial)
        starteta <- c(family$mu.linkfun(mean(mu)), family$sigma.linkfun(mean(sigma)))
      } else {
        eval(weight_mean_expression(family$mu.initial))
        eval(weight_mean_expression(family$sigma.initial))
        starteta <- c(family$mu.linkfun(weighted.mean(mu, weights)), family$sigma.linkfun(weighted.mean(sigma, weights)))
      }
      names(starteta) <- etanames
      return(starteta)
    }

    # define link function
    linkfun <- function(par){
      eta <- c(family$mu.linkfun(par[1]), family$sigma.linkfun(par[2]))
      names(eta) <- etanames
      return(eta)
    }

    # define function to get distribution parameters
    linkinv <- function(eta){
      par <- c(family$mu.linkinv(eta[1]), family$sigma.linkinv(eta[2]))
      names(par) <- parnames
      return(par)
    }

    # define functions that return inner derivatives as vector / matrix:
    dpardeta <- function(eta){
      return(c(dmdeta(eta), dddeta(eta)))
    }

    d2pardeta2 <- function(eta){
      return(c(d2mdeta2(eta), d2ddeta2(eta)))
    }


    # define functions that return outer derivatives as matrix / list of matrices:
    dldpar <- function(y, par){
      dmatrix <- cbind(dldm(y, par), dldd(y, par))
      return(dmatrix)
    }

    d2ldpar2 <- function(y, par){

      d2matrix <- rbind(cbind(d2ldm2(y, par), d2ldmdd(y, par)),
                        cbind(d2ldmdd(y, par), d2ldd2(y, par)))

      # d2matrix is of size (2*ny x 2)
      # transform to a list of matrices (length of the list equals the number of observations ny)
      # for each observation a matrix of size (2x2) is stored in d2list

      d2list <- list()
      ny <- NROW(y)
      length(d2list) <- ny
      for(i in 1:ny){
        d2list[[i]] <- d2matrix[c(i, ny+i),]
      }

      return(d2list)
    }


    # define p-, q- and r-function  (r-function not available for all GAMLSS families, e.g. rLOlc)
    pdist <- function(q, eta, lower.tail = TRUE, log.p = FALSE) {
      par <- linkinv(eta)
      do.call(get(paste0("p",family$family[1])),
              list(q, par[1], par[2],
                   lower.tail = lower.tail, log.p = log.p))
    }
    qdist <- function(p, eta, lower.tail = TRUE, log.p = FALSE) {
      par <- linkinv(eta)
      do.call(get(paste0("q",family$family[1])),
              list(p, par[1], par[2],
                   lower.tail = lower.tail, log.p = log.p))
    }
    rfun.available <- try(get(paste0("r",family$family[1])), silent = TRUE)
    if(inherits(rfun.available, "try-error")) {
      # warning("r-function not available for this family object")
      rdist <- NULL
    } else {
      rdist <- function(n, eta) {
        par <- linkinv(eta)
        do.call(get(paste0("r",family$family[1])), list(n, par[1], par[2]))
      }
    }
  }


  if(np == 3L){

    # define function for the calculation of initial values
    startfun <- function(y, weights = NULL) {
      mu <- sigma <- nu <-  NULL
      if(is.null(weights) || (length(weights)==0L)) {
        eval(family$mu.initial)
        eval(family$sigma.initial)
        eval(family$nu.initial)
        starteta <- c(family$mu.linkfun(mean(mu)), family$sigma.linkfun(mean(sigma)), family$nu.linkfun(mean(nu)))
      } else {
        eval(weight_mean_expression(family$mu.initial))
        eval(weight_mean_expression(family$sigma.initial))
        eval(weight_mean_expression(family$nu.initial))
        starteta <- c(family$mu.linkfun(weighted.mean(mu, weights)), family$sigma.linkfun(weighted.mean(sigma, weights)), family$nu.linkfun(weighted.mean(nu, weights)))
      }

      names(starteta) <- etanames
      return(starteta)
    }

    # define link function
    linkfun <- function(par){
      eta <- c(family$mu.linkfun(par[1]), family$sigma.linkfun(par[2]), family$nu.linkfun(par[3]))
      names(eta) <- etanames
      return(eta)
    }

    # define function to get distribution parameters
    linkinv <- function(eta){
      par <- c(family$mu.linkinv(eta[1]), family$sigma.linkinv(eta[2]), family$nu.linkinv(eta[3]))
      names(par) <- parnames
      return(par)
    }

    # define functions that return inner derivatives as vector / matrix:
    dpardeta <- function(eta){
      return(c(dmdeta(eta), dddeta(eta), dvdeta(eta)))
    }

    d2pardeta2 <- function(eta){
      return(c(d2mdeta2(eta), d2ddeta2(eta), d2vdeta2(eta)))
    }


    # define functions that return outer derivatives as matrix / list:
    dldpar <- function(y, par){
      dmatrix <- cbind(dldm(y, par), dldd(y, par), dldv(y, par))
      return(dmatrix)
    }

    d2ldpar2 <- function(y, par){

      d2matrix <- rbind(cbind(d2ldm2(y, par), d2ldmdd(y, par), d2ldmdv(y, par)),
                        cbind(d2ldmdd(y, par), d2ldd2(y, par), d2ldddv(y, par)),
                        cbind(d2ldmdv(y, par), d2ldddv(y, par), d2ldv2(y, par)))

      # d2matrix is of size (3*ny x 3)
      # transform to a list of matrices (length of the list equals the number of observations ny)
      # for each observation a matrix of size (3x3) is stored in d2list

      d2list <- list()
      ny <- NROW(y)
      length(d2list) <- ny
      for(i in 1:ny){
        d2list[[i]] <- d2matrix[c(i, ny+i, 2*ny+i),]
      }

      return(d2list)
    }


    # define p-, q- and r-function  (r-function not available for all GAMLSS families, e.g. rLOlc)
    pdist <- function(q, eta, lower.tail = TRUE, log.p = FALSE) {
      par <- linkinv(eta)
      do.call(get(paste0("p",family$family[1])),
              list(q, par[1], par[2], par[3],
                   lower.tail = lower.tail, log.p = log.p))
    }
    qdist <- function(p, eta, lower.tail = TRUE, log.p = FALSE) {
      par <- linkinv(eta)
      do.call(get(paste0("q",family$family[1])),
              list(p, par[1], par[2], par[3],
                   lower.tail = lower.tail, log.p = log.p))
    }
    rfun.available <- try(get(paste0("r",family$family[1])), silent = TRUE)
    if(inherits(rfun.available, "try-error")) {
      # warning("r-function not available for this family object")
      rdist <- NULL
    } else {
      rdist <- function(n, eta) {
        par <- linkinv(eta)
        do.call(get(paste0("r",family$family[1])), list(n, par[1], par[2], par[3]))
      }
    }
  }


  if(np == 4L){

    # define function for the calculation of initial values
    startfun <- function(y, weights = NULL) {
      mu <- sigma <- nu <- tau <- NULL
      if(is.null(weights) || (length(weights)==0L)) {
        eval(family$mu.initial)
        eval(family$sigma.initial)
        eval(family$nu.initial)
        eval(family$tau.initial)
        starteta <- c(family$mu.linkfun(mean(mu)), family$sigma.linkfun(mean(sigma)), family$nu.linkfun(mean(nu)), family$tau.linkfun(mean(tau)))
      } else {
        eval(weight_mean_expression(family$mu.initial))
        eval(weight_mean_expression(family$sigma.initial))
        eval(weight_mean_expression(family$nu.initial))
        eval(weight_mean_expression(family$tau.initial))
        starteta <- c(family$mu.linkfun(weighted.mean(mu, weights)), family$sigma.linkfun(weighted.mean(sigma, weights)), family$nu.linkfun(weighted.mean(nu, weights)), family$tau.linkfun(weighted.mean(tau, weights)))
      }

      names(starteta) <- etanames
      return(starteta)
    }

    # define link function
    linkfun <- function(par){
      eta <- c(family$mu.linkfun(par[1]), family$sigma.linkfun(par[2]), family$nu.linkfun(par[3]), family$tau.linkfun(par[4]))
      names(eta) <- etanames
      return(eta)
    }

    # define function to get distribution parameters
    linkinv <- function(eta){
      par <- c(family$mu.linkinv(eta[1]), family$sigma.linkinv(eta[2]), family$nu.linkinv(eta[3]), family$tau.linkinv(eta[4]))
      names(par) <- parnames
      return(par)
    }

    # define functions that return inner derivatives as vector / matrix:
    dpardeta <- function(eta){
      return(c(dmdeta(eta), dddeta(eta), dvdeta(eta), dtdeta(eta)))
    }

    d2pardeta2 <- function(eta){
      return(c(d2mdeta2(eta), d2ddeta2(eta), d2vdeta2(eta), d2tdeta2(eta)))
    }


    # define functions that return outer derivatives as matrix / list :
    dldpar <- function(y, par){
      dmatrix <- cbind(dldm(y, par), dldd(y, par), dldv(y, par), dldt(y, par))
      return(dmatrix)
    }

    d2ldpar2 <- function(y, par){

      d2matrix <- rbind(cbind(d2ldm2(y, par), d2ldmdd(y, par), d2ldmdv(y, par), d2ldmdt(y, par)),
                        cbind(d2ldmdd(y, par), d2ldd2(y, par), d2ldddv(y, par), d2ldddt(y, par)),
                        cbind(d2ldmdv(y, par), d2ldddv(y, par), d2ldv2(y, par), d2ldvdt(y, par)),
                        cbind(d2ldmdt(y, par), d2ldddt(y, par), d2ldvdt(y, par), d2ldt2(y, par)))

      # d2matrix is of size (4*ny x 4)
      # transform to a list of matrices (length of the list equals the number of observations ny)
      # for each observation a matrix of size (4x4) is stored in d2list

      d2list <- list()
      ny <- NROW(y)
      length(d2list) <- ny
      for(i in 1:ny){
        d2list[[i]] <- d2matrix[c(i, ny+i, 2*ny+i, 3*ny+i),]
      }

      return(d2list)
    }


    # define p-, q- and r-function  (r-function not available for all GAMLSS families, e.g. rLOlc)
    pdist <- function(q, eta, lower.tail = TRUE, log.p = FALSE) {
      par <- linkinv(eta)
      do.call(get(paste0("p",family$family[1])),
              list(q, par[1], par[2], par[3], par[4],
                   lower.tail = lower.tail, log.p = log.p))
    }
    qdist <- function(p, eta, lower.tail = TRUE, log.p = FALSE) {
      par <- linkinv(eta)
      do.call(get(paste0("q",family$family[1])),
              list(p, par[1], par[2], par[3], par[4],
                   lower.tail = lower.tail, log.p = log.p))
    }
    rfun.available <- try(get(paste0("r",family$family[1])), silent = TRUE)
    if(inherits(rfun.available, "try-error")) {
      # warning("r-function not available for this family object")
      rdist <- NULL
    } else {
      rdist <- function(n, eta) {
        par <- linkinv(eta)
        do.call(get(paste0("r",family$family[1])), list(n, par[1], par[2], par[3], par[4]))
      }
    }
  }




  ddist <- function(y, eta, log = TRUE, weights = NULL, sum = FALSE) {
    par <- linkinv(eta)
    input <- list()

    # fixed parameters do not need to be added here (for nu = 0 and c(nu, tau) = c(1.5, 2) respectively),
    # because in the density function dLNO and dNET nu (and tau) are by default set to to c(0) or c(1.5, 2) respectively
    #  => following if-conditions only necessary for different values for nu (and tau)
    if(family$nopar != np) {
      if(family$family[1] == "LNO") {
        par <- c(par, 0)
        parnames <- c(parnames, "nu")
      }
      if(family$family[1] == "NET") {
        par <- c(par, 1.5, 2)
        parnames <- c(parnames, "nu", "tau")
      }
    }

    inputnames <- c("x", parnames, "log")
    input[[1]] <- y
    input <- c(input, par)                          # <- rep.int(par[i-1], length(y))   (FIX?)
    if(any(family$family%in%.distfit.bi.list)) {
      input <- c(input, bd)      # additional parameter bd (binomial denominator for families in .distfit.bi.list)
      inputnames <- c("x", parnames, "bd", "log")
    }
    input <- c(input, log)
    names(input) <- inputnames
    eval <- do.call(get(paste0("d", family$family[[1]])), input)
    if(sum) {
      if(is.null(weights) || (length(weights)==0L)) {
        ny <- NROW(y)
        weights <- rep.int(1, ny)
      }
      eval <- sum(weights * eval)
    }
    return(eval)
  }


  ## score / estfun (first-order partial derivatives of the (positive) log-likelihood function)
  sdist <- function(y, eta, weights = NULL, sum = FALSE) {
    par <- linkinv(eta)

    if(family$nopar != np){
      if(family$family[1] == "LNO") {
        par <- c(par, 0)
        eta <- c(eta, 0)
      }
      if(family$family[1] == "NET") {
        par <- c(par, 1.5, 2)
        eta <- c(eta, 1.5, 2)
      }
    }

    score <- t(t(dldpar(y, par)) * dpardeta(eta))
    score <- as.matrix(score)
    colnames(score) <- etanames
    if(sum) {
      ny <- NROW(y)
      if(is.null(weights) || (length(weights)==0L)) weights <- rep.int(1, ny)
      score <- colSums(weights * score)
    }
    return(score)
  }


  ## hessian (second-order partial derivatives of the (positive) log-likelihood function)
  hdist <- function(y, eta, weights = NULL) {
    ny <- NROW(y)
    if(is.null(weights) || (length(weights)==0L)) weights <- rep.int(1, ny)

    par <- linkinv(eta)

    if(family$nopar != np){
      if(family$family[1] == "LNO") {
        par <- c(par, 0)
        eta <- c(eta, 0)
      }
      if(family$family[1] == "NET") {
        par <- c(par, 1.5, 2)
        eta <- c(eta, 1.5, 2)
      }
    }

    ## calculate derivative vectors / matrices / lists
    d2ldpar2.list <- d2ldpar2(y, par)
    dldpar.mat <- dldpar(y, par)

    dpardeta.vec <- dpardeta(eta)
    d2pardeta2.vec <- d2pardeta2(eta)

    ## calculation is split up in 2 parts:
    # second outer derivatives times first inner derivatives and a diagonal matrix with the first outer and the second inner derivatives
    hess.list <- list()
    length(hess.list) <- length(d2ldpar2.list)
    for(i in 1:ny){
      hess.list[[i]] <- weights[i] * (t(d2ldpar2.list[[i]] * dpardeta.vec) * dpardeta.vec + diag(np) * as.vector(dldpar.mat[i,]) * d2pardeta2.vec)
    }

    ## calculate the sum over all matrices in the list (each for one observation)
    hess <- Reduce('+', hess.list)
    hess <- as.matrix(hess)
    colnames(hess) <- rownames(hess) <-  etanames
    return(hess)
  }

  link <- linknames      # as defined above (within if(np == ))

  linkinvdr <- dpardeta

  mle <- FALSE

  dist_list <- list(family.name = paste(family$family[2], "Distribution", sep = " "),
                    ddist = ddist,
                    sdist = sdist,
                    hdist = hdist,
                    pdist = pdist,
                    qdist = qdist,
                    rdist = rdist,
                    link = link,
                    linkfun = linkfun,
                    linkinv = linkinv,
                    linkinvdr = linkinvdr,
                    startfun = startfun,
                    mle = mle,
                    gamlssobj = TRUE,
                    censored = censored,
                    censtype = censtype,
                    censpoint = censpoint
  )
}